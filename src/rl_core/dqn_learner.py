"""``DQNLearner`` — the Double DQN update loop and its update-keyed cadences.

Takes batches from ``batch_queue`` (drawn by ``ReplayReader``, shaped by the
project), runs one Double DQN step per batch, hands ``(indices, |td|)`` to the
``PriorityUpdater``, asks the ``QTargetWorker`` for target values and sends it
weight syncs, decays the learning rate, and queues snapshots and latest
checkpoints for the ``CheckpointWriter``. Stops on ``shutdown_event``, a
sentinel, or ``max_updates`` (0 = no cap), and sets ``shutdown_event`` on exit.

The project supplies the network, the optimizer and two hooks: ``prepare``
turns a queued batch into a ``DQNBatch`` on the learner device, and
``make_target_request`` builds the ``QTargetWorker`` request for its next
states. Optional hooks: ``release`` (after every batch, e.g. recycle a pinned
buffer), ``clip_next_q`` (bound the target values) and ``on_step``
(diagnostics).

Step::

    next_action = argmax_a q_online(s')[a]
    next_q      = q_target(s')[next_action]                 (optionally clipped)
    target      = γ^(τ-1) · (r + γ · next_q · 1[¬done])     (τ ≡ 1 without durations)
    loss        = mean(IS_weight · Huber(q_online(s)[a], target))

Only ``optimizer.step()`` and the weight clones take ``network_lock``; forward
and backward run without it, as do the ``QNetworkWorker`` reads.
"""

import logging
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch
from ai_runtime.thread_stages import SHUTDOWN_SENTINEL
from torch import nn
from torch.nn import functional

from rl_core.dqn_batch import DQNBatch
from rl_core.events.checkpoint_request import CheckpointRequest
from rl_core.events.latest_request import LatestRequest
from rl_core.protocols.learner_telemetry_protocol import LearnerTelemetryProtocol
from rl_core.sync_request import SyncRequest

logger = logging.getLogger(__name__)


class DQNLearner:
    """Learner thread: batch → Double DQN step → priorities; then the cadences."""

    # Batch-queue poll interval; bounds shutdown latency while starved.
    _QUEUE_POLL_TIMEOUT: float = 0.1
    # Push-ratio throttle re-check interval.
    _THROTTLE_POLL_SECONDS: float = 0.05
    # Bounded-put retry interval for the priority queue.
    _QUEUE_PUT_TIMEOUT: float = 0.5
    # Target-sync INFO log interval, in updates.
    _SYNC_LOG_INTERVAL: int = 1000

    def __init__(
        self,
        q_online: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_queue: queue.Queue,
        target_queue: queue.Queue,
        priority_queue: queue.Queue,
        write_queue: queue.Queue | None,
        telemetry: LearnerTelemetryProtocol,
        network_lock: threading.Lock,
        shutdown_event: threading.Event,
        *,
        prepare: Callable[[Any], DQNBatch],
        make_target_request: Callable[[Any, Future], Any],
        gamma: float,
        lr_init: float,
        lr_min: float,
        lr_decay: float = 1.0,
        lr_schedule: str = "exponential",
        max_updates: int = 0,
        target_sync_every: int,
        checkpoint_every: int = 0,
        min_checkpoint_update: int = 0,
        latest_every: int = 0,
        push_per_update_ratio: float = 0.0,
        layout_dims: dict | None = None,
        state_builder: Callable[[], Any] | None = None,
        release: Callable[[Any], None] | None = None,
        clip_next_q: Callable[[torch.Tensor], torch.Tensor] | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> None:
        """Bind the network, optimizer, queues, hooks and cadences.

        ``lr_schedule`` is ``"exponential"`` (× ``lr_decay`` per update) or
        ``"linear"`` (``lr_init`` → ``lr_min`` over ``max_updates``), floored at
        ``lr_min``. ``checkpoint_every`` / ``latest_every`` of 0, or no
        ``write_queue``, turn snapshots / latest checkpoints off; ``state_builder``
        assembles the latest-checkpoint state.
        """
        if lr_schedule not in ("exponential", "linear"):
            raise ValueError(f"lr_schedule must be exponential or linear; got {lr_schedule!r}")
        if lr_schedule == "linear" and max_updates <= 0:
            raise ValueError("a linear lr_schedule needs max_updates > 0")
        if target_sync_every < 1:
            raise ValueError(f"target_sync_every must be >= 1; got {target_sync_every}")
        self._q_online = q_online
        self._optimizer = optimizer
        self._batch_queue = batch_queue
        self._target_queue = target_queue
        self._priority_queue = priority_queue
        self._write_queue = write_queue
        self._telemetry = telemetry
        self._network_lock = network_lock
        self._shutdown_event = shutdown_event
        self._prepare = prepare
        self._make_target_request = make_target_request
        self._gamma = float(gamma)
        self._lr_init = float(lr_init)
        self._lr_min = float(lr_min)
        self._lr_decay = float(lr_decay)
        self._lr_schedule = lr_schedule
        self._max_updates = int(max_updates)
        self._target_sync_every = int(target_sync_every)
        self._checkpoint_every = int(checkpoint_every)
        self._min_checkpoint_update = int(min_checkpoint_update)
        self._latest_every = int(latest_every)
        self._push_per_update_ratio = float(push_per_update_ratio)
        self._layout_dims = dict(layout_dims or {})
        self._state_builder = state_builder
        self._release = release
        self._clip_next_q = clip_next_q
        self._on_step = on_step
        self._device = next(q_online.parameters()).device

        self._update_count: int = 0
        self._last_target_sync_update: int = 0
        # Cursor for the rate-limited sync log (not checkpointed).
        self._last_sync_log_update: int = 0
        self._last_snapshot: int = 0

    @property
    def updates(self) -> int:
        """Updates done."""
        return self._update_count

    def run(self) -> None:
        """Learner thread main loop; sets ``shutdown_event`` on exit."""
        ratio = self._push_per_update_ratio
        logger.info("DQNLearner.run started; waiting for first batch ...")
        while not self._shutdown_event.is_set():
            # Updates must not outrun production.
            if ratio > 0 and self._update_count * ratio > self._telemetry.transitions_pushed:
                self._shutdown_event.wait(self._THROTTLE_POLL_SECONDS)
                continue
            try:
                batch = self._batch_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if batch is SHUTDOWN_SENTINEL:
                break
            logger.debug("learner: batch received")
            try:
                self._step(batch)
            except Exception as exc:
                logger.exception("DQNLearner.run: DQN step failed (%s); continuing", exc)
                continue
            finally:
                if self._release is not None:
                    self._release(batch)

            self._decay_lr()
            self._sync_target()
            self._write_latest_checkpoint()
            self._maybe_write_snapshot()

            if 0 < self._max_updates <= self._update_count:
                logger.info("DQNLearner.run: max_updates=%d reached; stopping",
                            self._max_updates)
                break

        logger.info("DQNLearner.run exiting: %d updates", self._update_count)
        # A recoverable artefact even if the cadence never fired.
        self._write_final_checkpoint()
        self._shutdown_event.set()

    def _step(self, raw) -> None:
        """One Double DQN update: target request first, online forwards, step."""
        # The target thread starts on s' while this thread works below.
        target_future: Future = Future()
        self._target_queue.put(self._make_target_request(raw, target_future))

        b = self._prepare(raw)
        logger.debug("learner step: prepared")

        q_values = self._q_online(b.states)                                   # (B, A)
        q_taken = q_values.gather(1, b.actions.unsqueeze(1)).squeeze(1)       # (B,)

        with torch.no_grad():
            next_actions = self._q_online(b.next_states).argmax(dim=1)        # (B,)
            logger.debug("learner step: online forwards")
            next_q_target = target_future.result()                            # (B, A)
            logger.debug("learner step: target ready")
            next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            if self._clip_next_q is not None:
                next_q_value = self._clip_next_q(next_q_value)
            next_state_value = next_q_value * (~b.dones).float()
            gamma = self._gamma
            if b.durations is None:
                target = b.rewards + gamma * next_state_value
            else:
                # SMDP: reward discounted by γ^(τ-1), successor by γ^τ.
                target = torch.pow(gamma, b.durations - 1.0) * (
                    b.rewards + gamma * next_state_value)

        td_error = (target - q_taken).detach()
        loss_per_sample = functional.smooth_l1_loss(q_taken, target, reduction="none")
        loss = (loss_per_sample * b.is_weights).mean()

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        with self._network_lock:
            self._optimizer.step()
        logger.debug("learner step: stepped")

        td_errors_abs = td_error.abs().cpu().numpy()   # device sync barrier
        self._put_priorities(b.indices, td_errors_abs)

        if self._on_step is not None:
            self._on_step({
                "update": self._update_count, "loss": loss, "td_error": td_error,
                "q_values": q_values, "q_taken": q_taken, "target": target,
                "next_q_value": next_q_value, "dones": b.dones,
                "lr": self._optimizer.param_groups[0]["lr"],
            })

        self._update_count += 1
        self._telemetry.note_update(self._update_count)

    def _put_priorities(self, indices, td_errors_abs) -> None:
        """Bounded put onto the priority queue, staying shutdown-responsive."""
        while not self._shutdown_event.is_set():
            try:
                self._priority_queue.put((indices, td_errors_abs),
                                         timeout=self._QUEUE_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

    def _decay_lr(self) -> None:
        """One learning-rate step, floored at ``lr_min``."""
        if self._lr_schedule == "linear":
            frac = min(1.0, self._update_count / self._max_updates)
            lr = self._lr_init - (self._lr_init - self._lr_min) * frac
        else:
            lr = self._optimizer.param_groups[0]["lr"] * self._lr_decay
        lr = max(lr, self._lr_min)
        for pg in self._optimizer.param_groups:
            pg["lr"] = lr

    def _sync_target(self) -> None:
        """Every ``target_sync_every`` updates, send a cloned ``q_online`` state."""
        if self._update_count < self._last_target_sync_update + self._target_sync_every:
            return
        with self._network_lock:
            snapshot = {k: v.detach().clone() for k, v in self._q_online.state_dict().items()}
        self._target_queue.put(SyncRequest(state_dict=snapshot))
        self._last_target_sync_update = self._update_count
        if self._update_count >= self._last_sync_log_update + self._SYNC_LOG_INTERVAL:
            logger.info("DQNLearner: target sync requested at update %d", self._update_count)
            self._last_sync_log_update = self._update_count

    def _maybe_write_snapshot(self) -> None:
        """Past ``min_checkpoint_update``, every ``checkpoint_every`` updates: a CPU clone."""
        if self._write_queue is None or self._checkpoint_every <= 0:
            return
        if self._update_count < self._min_checkpoint_update:
            return
        if self._update_count < self._last_snapshot + self._checkpoint_every:
            return
        with self._network_lock:
            state_dict = {k: v.detach().cpu().clone()
                          for k, v in self._q_online.state_dict().items()}
        self._write_queue.put(CheckpointRequest(
            update=self._update_count, state_dict=state_dict, dims=self._layout_dims,
        ))
        self._last_snapshot = self._update_count

    def _write_latest_checkpoint(self) -> None:
        """Every ``latest_every`` updates, queue the ``state_builder`` state."""
        if self._write_queue is None or self._latest_every <= 0:
            return
        if self._update_count == 0 or self._update_count % self._latest_every != 0:
            return
        if self._state_builder is None:
            logger.debug("Latest checkpoint cadence reached but no state_builder; skipping")
            return
        self._write_queue.put(LatestRequest(state=self._state_builder()))
        logger.info("DQNLearner: latest checkpoint queued at update %d", self._update_count)

    def _write_final_checkpoint(self) -> None:
        """One latest checkpoint at exit, regardless of cadence."""
        if self._write_queue is None or self._state_builder is None:
            return
        self._write_queue.put(LatestRequest(state=self._state_builder()))
        logger.info("DQNLearner: final latest_checkpoint queued at exit")

    def get_state(self) -> dict:
        """Optimizer state and the update-keyed schedule cursors, for a checkpoint."""
        return {
            "optimizer_state_dict": self._optimizer.state_dict(),
            "update_count": int(self._update_count),
            "last_target_sync_update": int(self._last_target_sync_update),
            "last_snapshot": int(self._last_snapshot),
        }

    def load_state(self, state: dict) -> None:
        """Restore ``get_state`` output; republishes the update count."""
        self._optimizer.load_state_dict(state["optimizer_state_dict"])
        self._update_count = int(state["update_count"])
        self._last_target_sync_update = int(state.get("last_target_sync_update", 0))
        self._last_snapshot = int(state.get("last_snapshot", 0))
        self._telemetry.note_update(self._update_count)
