"""``DDPGLearner`` — the DDPG critic update loop and its update-keyed cadences.

Takes batches from ``batch_queue``, lets the subclass run the critic's forward and backward pass
(``_critic_backward``: gradients accumulated, one ``CriticStep`` returned), clips the gradient norm, steps the
optimizer, moves the target critic toward the critic, hands ``(indices, |td|)`` to the ``PriorityUpdater``, lowers
the learning rate linearly over ``max_updates``, and queues critic snapshots for the ``CheckpointWriter``.

Step::

    target critic θ' ← (1 − τ) · θ' + τ · θ        after every optimizer step

Stops on ``shutdown_event``, a sentinel, or ``max_updates``; ``done`` is set once the final snapshot is queued.
A failure sets ``shutdown_event`` and re-raises. With a ``GradientExchange``, one learner per critic copy averages
its gradients with the others before the clip; only rank 0 (the primary) reports updates to the telemetry.
"""

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Any

import torch
from ai_runtime.thread_stages import SHUTDOWN_SENTINEL
from torch import nn

from rl_core.critic_step import CriticStep
from rl_core.events.checkpoint_request import CheckpointRequest
from rl_core.gradient_exchange import GradientExchange
from rl_core.protocols.learner_telemetry_protocol import LearnerTelemetryProtocol

logger = logging.getLogger(__name__)


class DDPGLearner(ABC):
    """Learner thread: batch → critic step → priorities, target update; then the cadences."""

    # Batch-queue poll interval; bounds shutdown latency while starved.
    _QUEUE_POLL_TIMEOUT: float = 0.1
    # Push-ratio throttle re-check interval.
    _THROTTLE_POLL_SECONDS: float = 0.05
    # Bounded-put retry interval for the priority queue.
    _QUEUE_PUT_TIMEOUT: float = 0.5

    def __init__(
        self,
        critic: nn.Module,
        critic_target: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_queue: queue.Queue,
        priority_queue: queue.Queue,
        write_queue: queue.Queue | None,
        telemetry: LearnerTelemetryProtocol,
        shutdown_event: threading.Event,
        *,
        tau: float,
        lr_init: float,
        lr_min: float,
        max_updates: int,
        grad_clip: float,
        checkpoint_every: int = 0,
        push_per_update_ratio: float = 0.0,
        snapshot_dims: dict | None = None,
        exchange: GradientExchange | None = None,
        rank: int = 0,
    ) -> None:
        """Bind the two critics, the optimizer, the queues and the cadences.

        ``lr`` falls linearly from ``lr_init`` to ``lr_min`` over ``max_updates``. ``checkpoint_every`` of 0, or no
        ``write_queue``, turns snapshots off. ``push_per_update_ratio`` > 0 holds an update back until that many
        transitions per update are stored. ``exchange`` and ``rank`` place this learner among several copies.
        """
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1]; got {tau}")
        if max_updates < 1:
            raise ValueError(f"max_updates must be >= 1; got {max_updates}")
        if grad_clip <= 0.0:
            raise ValueError(f"grad_clip must be > 0; got {grad_clip}")
        self._critic = critic
        self._critic_target = critic_target
        self._optimizer = optimizer
        self._batch_queue = batch_queue
        self._priority_queue = priority_queue
        self._write_queue = write_queue
        self._telemetry = telemetry
        self._shutdown_event = shutdown_event
        self._tau = float(tau)
        self._lr_init = float(lr_init)
        self._lr_min = float(lr_min)
        self._max_updates = int(max_updates)
        self._grad_clip = float(grad_clip)
        self._checkpoint_every = int(checkpoint_every)
        self._push_per_update_ratio = float(push_per_update_ratio)
        self._snapshot_dims = dict(snapshot_dims or {})
        self._exchange = exchange
        self._rank = int(rank)
        self._update_count: int = 0
        self._last_snapshot: int = 0
        self.done = threading.Event()

    @property
    def updates(self) -> int:
        """Updates done."""
        return self._update_count

    @property
    def primary(self) -> bool:
        """Whether this learner is rank 0, the one that reports updates."""
        return self._rank == 0

    @abstractmethod
    def _critic_backward(self, batch: Any) -> CriticStep:
        """Forward and backward pass of the critic on one batch; gradients accumulate, no optimizer step."""

    def _release(self, batch: Any) -> None:
        """After every batch, whatever the outcome; nothing by default."""

    def _on_step(self, update: int, metrics: dict[str, float]) -> None:
        """Diagnostics after every update; nothing by default."""

    def _snapshot_state(self) -> dict:
        """The critic's weights as a CPU copy, for a snapshot."""
        return {name: tensor.detach().cpu().clone() for name, tensor in self._critic.state_dict().items()}

    def run(self) -> None:
        """Learner thread main loop; a failure sets ``shutdown_event`` and re-raises."""
        logger.info("DDPGLearner.run started; max_updates=%d", self._max_updates)
        try:
            while not self._shutdown_event.is_set():
                if self._throttled():
                    self._shutdown_event.wait(self._THROTTLE_POLL_SECONDS)
                    continue
                try:
                    batch = self._batch_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                if batch is SHUTDOWN_SENTINEL:
                    break
                try:
                    stepped = self._step(batch)
                finally:
                    self._release(batch)
                if not stepped:
                    break
                self._maybe_write_snapshot()
                if self._update_count >= self._max_updates:
                    logger.info("DDPGLearner.run: max_updates=%d reached; stopping", self._max_updates)
                    break
            self._write_final_snapshot()
        except Exception:
            self._shutdown_event.set()
            raise
        finally:
            self.done.set()
        logger.info("DDPGLearner.run exiting: %d updates", self._update_count)

    def _throttled(self) -> bool:
        """Whether the next update would outrun production."""
        ratio = self._push_per_update_ratio
        return ratio > 0 and (self._update_count + 1) * ratio > self._telemetry.transitions_pushed

    def _step(self, batch: Any) -> bool:
        """One critic update: backward, exchange, clip, step, target update, priorities; False on shutdown."""
        self._set_lr()
        self._optimizer.zero_grad(set_to_none=True)
        result = self._critic_backward(batch)
        if self._exchange is not None and not self._exchange.average(self._rank):
            return False
        grad_norm = float(nn.utils.clip_grad_norm_(self._critic.parameters(), self._grad_clip))
        self._optimizer.step()
        self._update_target()
        self._update_count += 1
        if self.primary:
            self._telemetry.note_update(self._update_count)
        self._put_priorities(result)
        self._on_step(self._update_count, {
            **result.metrics, "grad_norm": grad_norm, "lr": self._optimizer.param_groups[0]["lr"]})
        return True

    def _set_lr(self) -> None:
        """The learning rate of this update: linear from ``lr_init`` to ``lr_min`` over ``max_updates``."""
        frac = min(1.0, self._update_count / self._max_updates)
        lr = self._lr_init - (self._lr_init - self._lr_min) * frac
        for group in self._optimizer.param_groups:
            group["lr"] = lr

    def _update_target(self) -> None:
        """θ' ← (1 − τ) · θ' + τ · θ, parameter by parameter."""
        with torch.no_grad():
            for target, online in zip(self._critic_target.parameters(), self._critic.parameters()):
                target.lerp_(online.to(target.dtype), self._tau)

    def _put_priorities(self, result: CriticStep) -> None:
        """Bounded put onto the priority queue, staying shutdown-responsive."""
        while not self._shutdown_event.is_set():
            try:
                self._priority_queue.put((result.indices, result.td_abs), timeout=self._QUEUE_PUT_TIMEOUT)
                return
            except queue.Full:
                continue

    def _maybe_write_snapshot(self) -> None:
        """Every ``checkpoint_every`` updates, a CPU copy of the critic for the writer."""
        if self._write_queue is None or self._checkpoint_every <= 0:
            return
        if self._update_count < self._last_snapshot + self._checkpoint_every:
            return
        self._queue_snapshot(self._write_queue)

    def _write_final_snapshot(self) -> None:
        """One snapshot at exit when updates were done since the last one."""
        if self._write_queue is None or self._checkpoint_every <= 0:
            return
        if self._update_count > self._last_snapshot:
            self._queue_snapshot(self._write_queue)

    def _queue_snapshot(self, write_queue: queue.Queue) -> None:
        """Queue the snapshot at the current update; waits for room in the write queue."""
        write_queue.put(CheckpointRequest(
            update=self._update_count, state_dict=self._snapshot_state(), dims=self._snapshot_dims))
        self._last_snapshot = self._update_count
        logger.info("DDPGLearner: snapshot queued at update %d", self._update_count)
