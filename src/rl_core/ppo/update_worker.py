"""``PPOUpdateWorker`` — the thread that owns the policy and trains it.

Clipped surrogate with a value term and an entropy bonus:
L = -min(r*A, clip(r, 1-e, 1+e)*A) + c_v*(V - R)^2 - c_e*H, r = exp(lp - lp_old).
On-policy, so the collect side must ``wait_idle()`` before gathering the next
rollout — the stored log-probs are only valid against the weights that made them.
"""

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as torch_f

from rl_core.ppo.network import PPOPolicyNetwork
from rl_core.ppo.rollout_batch import PPORolloutBatch

logger = logging.getLogger(__name__)


class PPOUpdateWorker(threading.Thread):
    """Drains the rollout queue: one clipped-surrogate update per batch."""

    # How long ``run()`` blocks on the queue between shutdown checks.
    _QUEUE_POLL_TIMEOUT: float = 0.1
    # Floor on the advantage spread before normalizing.
    _EPS: float = 1e-8
    # Below this spread a minibatch is treated as carrying no ranking signal;
    # 1e-8 is far too small here, where most episodes score identically.
    _MIN_SPREAD: float = 1e-3
    # Ceiling on |log ratio| before exponentiating; e^20 is already far
    # outside anything the clipped surrogate can use.
    _MAX_LOG_RATIO: float = 20.0

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout_queue: queue.Queue,
        device: torch.device,
        shutdown_event: threading.Event,
        epochs: int = 4,
        minibatch_size: int = 256,
        clip_range: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 0.5,
    ) -> None:
        """Bind the policy and its optimizer; the rest are the PPO knobs."""
        super().__init__(name="ppo-update-worker", daemon=True)
        self.model = model
        self.optimizer = optimizer
        self._rollout_queue = rollout_queue
        self._device = device
        self._shutdown_event = shutdown_event
        self._epochs = int(epochs)
        self._minibatch_size = int(minibatch_size)
        self._clip_range = float(clip_range)
        self._value_coefficient = float(value_coefficient)
        self._entropy_coefficient = float(entropy_coefficient)
        self._max_grad_norm = float(max_grad_norm)
        # One rollout is in flight at a time (on-policy), so completion is a
        # token on a queue rather than counters behind a condition.
        self._done_queue: queue.Queue = queue.Queue()
        self._done = 0
        self._error: Optional[BaseException] = None
        self._updates = 0
        self._sums: dict[str, float] = {}

    @property
    def updates(self) -> int:
        """Gradient steps taken since construction."""
        return self._updates

    def submit(self, batch: PPORolloutBatch) -> None:
        """Queue one rollout; blocks while the queue is full."""
        self._raise_if_failed()
        self._rollout_queue.put(batch)

    def run(self) -> None:
        """Take rollouts until shutdown and the queue is empty; failures end the thread loudly."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    batch = self._rollout_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                except queue.Empty:
                    continue
                stats = self.step(batch)
                self._done += 1
                self._done_queue.put(self._done)
                logger.info(
                    "ppo update %d: loss %.4f (policy %.4f, value %.4f), "
                    "entropy %.4f, grad %.3f, ratio@epoch0 %.6f, "
                    "clipped %.3f, %d steps, %.2fs",
                    self._done, stats["loss"], stats["policy_loss"],
                    stats["value_loss"], stats["entropy"], stats["grad_norm"],
                    stats["ratio_epoch0"], stats["clipped_fraction"],
                    stats["grad_steps"], stats["seconds"],
                )
        except BaseException as exc:
            self._error = exc
            self._shutdown_event.set()
            self._done_queue.put(None)   # wake a waiter so it sees the failure
            logger.exception("ppo update worker failed")
            raise

    def step(self, batch: PPORolloutBatch) -> dict:
        """Several epochs of minibatch updates over one rollout."""
        started = time.monotonic()
        before = self._updates
        tensors = self._to_device(batch)
        count = len(batch)
        order = np.arange(count)
        rng = np.random.default_rng(self._updates)
        totals: dict[str, float] = {}
        first_ratio: float | None = None
        for epoch in range(self._epochs):
            rng.shuffle(order)
            for start in range(0, count, self._minibatch_size):
                index = torch.as_tensor(order[start:start + self._minibatch_size],
                                        dtype=torch.long, device=self._device)
                stats = self._minibatch(tensors, index)
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + value
                totals["batches"] = totals.get("batches", 0.0) + 1.0
                if first_ratio is None:
                    first_ratio = float(stats["ratio"])
        # Leave the shared policy in eval mode: the rollout threads read it
        # between updates, and their log-probs must be the deterministic ones.
        self.model.eval()
        batches = max(totals.pop("batches", 1.0), 1.0)
        out = {key: value / batches for key, value in totals.items()}
        # The first minibatch is the only one taken at the weights that
        # produced the stored log-probs, so this must read 1.0 exactly
        # whatever the minibatch size; later ones have already moved.
        out["ratio_epoch0"] = first_ratio if first_ratio is not None else 0.0
        out["grad_steps"] = self._updates - before
        out["seconds"] = round(time.monotonic() - started, 2)
        for key, value in out.items():
            self._sums[key] = self._sums.get(key, 0.0) + value
        return out

    def _minibatch(self, tensors: dict[str, Tensor], index: Tensor) -> dict:
        """One gradient step on the sampled transitions."""
        self.model.train()
        return self._gradient_step(self.model(tensors["observations"][index]),
                                   tensors, index)

    def _gradient_step(self, out: Tensor, tensors: dict[str, Tensor],
                       index: Tensor) -> dict:
        """The clipped-surrogate loss on a forward's ``out``, then one optimizer step.

        ``out`` holds the rows of ``index``, in order. Shared by every update worker;
        only how ``out`` is produced differs.
        """
        actions = tensors["actions"][index]
        invalid = tensors["invalid"][index]
        old_log_probs = tensors["log_probs"][index]
        advantages = tensors["advantages"][index]
        returns = tensors["returns"][index]
        logits, values = PPOPolicyNetwork.split(out)
        log_probs = PPOPolicyNetwork.masked_log_probs(logits, invalid)
        taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Clamp before exponentiating. The masked log-softmax floors invalid
        # actions near log(MASKED_LOGIT), so a log-ratio can reach the
        # hundreds and overflow to inf; inf times a zero advantage is NaN.
        # Clipping bounds the surrogate to
        # 1 +/- clip_range anyway, so nothing beyond this range is used.
        ratio = (taken - old_log_probs).clamp(-self._MAX_LOG_RATIO,
                                              self._MAX_LOG_RATIO).exp()

        # A minibatch whose episodes all scored the same carries no ranking
        # information, and its advantage spread is near zero — dividing by it
        # amplifies the policy gradient without bound. Leave such a batch
        # unnormalized; its centred advantages are ~0, so it contributes
        # nothing rather than exploding.
        centred = advantages - advantages.mean()
        spread = advantages.std()
        normalized = (centred / spread if spread > self._MIN_SPREAD
                      else centred)
        unclipped = ratio * normalized
        clipped = ratio.clamp(1.0 - self._clip_range,
                              1.0 + self._clip_range) * normalized
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = torch_f.mse_loss(values, returns)
        entropy = PPOPolicyNetwork.entropy(log_probs, invalid).mean()
        loss = (policy_loss
                + self._value_coefficient * value_loss
                - self._entropy_coefficient * entropy)

        # Stop the moment the objective goes non-finite: once the weights are
        # NaN every later rollout is noise, and the masked log-probs silently
        # degrade to a uniform draw over invalid actions.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at update {self._updates}: policy "
                f"{float(policy_loss):.4g}, value {float(value_loss):.4g}, "
                f"entropy {float(entropy):.4g}; returns span "
                f"[{float(returns.min()):.4g}, {float(returns.max()):.4g}]")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self._max_grad_norm)
        self.optimizer.step()
        self._updates += 1
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "grad_norm": float(grad_norm),
            "ratio": float(ratio.detach().mean()),
            "clipped_fraction": float(
                ((ratio - 1.0).abs() > self._clip_range).float().mean()),
        }

    def _to_device(self, batch: PPORolloutBatch) -> dict[str, Tensor]:
        """One host-to-device move per column; the rollout arrives as numpy."""
        return {
            "observations": torch.as_tensor(batch.observations, dtype=torch.float32,
                                            device=self._device),
            "actions": torch.as_tensor(batch.actions, dtype=torch.long,
                                       device=self._device),
            "invalid": torch.as_tensor(batch.invalid, dtype=torch.bool,
                                       device=self._device),
            "log_probs": torch.as_tensor(batch.log_probs, dtype=torch.float32,
                                         device=self._device),
            "advantages": torch.as_tensor(batch.advantages, dtype=torch.float32,
                                          device=self._device),
            "returns": torch.as_tensor(batch.returns, dtype=torch.float32,
                                       device=self._device),
        }

    def epoch_stats(self) -> dict:
        """Mean of every recorded quantity since the last call."""
        n = max(self._done, 1)
        stats = {key: value / n for key, value in self._sums.items()}
        stats["updates"] = self._updates
        self._sums = {}
        return stats

    def wait_idle(self) -> None:
        """Block until the submitted rollout has been trained on."""
        while True:
            try:
                self._done_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
                break
            except queue.Empty:
                self._raise_if_failed()
                if not self.is_alive():
                    raise RuntimeError("ppo update worker exited")
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        """Re-raise the worker's failure in the caller's thread."""
        if self._error is not None:
            raise RuntimeError("ppo update worker failed") from self._error
