"""``PPOTrainer`` — the on-policy loop: collect, advantages, update, wait.

Runs on the caller's thread and owns the ``PPOUpdateWorker`` lifecycle: starts
it, hands it each rollout, waits for the update (the next rollout must come
from the new weights), and on exit sets ``shutdown_event`` and joins it.
Stops on the timestep budget, ``max_updates`` (0 = no cap), the shutdown
event, or ``after_update`` returning True.

The project supplies ``collect`` (one rollout from the current policy) and
optional hooks: ``report`` (before each update), ``after_update`` (scoring,
checkpoints, early stop) and ``finish`` (after the loop, before the join).
"""

import logging
import threading
from collections.abc import Callable

from rl_core.ppo.rollout_batch import PPORolloutBatch
from rl_core.ppo.update_worker import PPOUpdateWorker

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Collect → GAE → update → wait, until a stop; then the project's finish."""

    def __init__(
        self,
        worker: PPOUpdateWorker,
        shutdown_event: threading.Event,
        *,
        collect: Callable[[], PPORolloutBatch],
        timesteps: int,
        max_updates: int = 0,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        report: "Callable[[int, PPORolloutBatch, int], None] | None" = None,
        after_update: "Callable[[int], bool] | None" = None,
        finish: "Callable[[int, int], None] | None" = None,
        join_timeout: float = 60.0,
    ) -> None:
        """Bind the update worker and the hooks; ``report`` gets (index, rollout, steps)."""
        self._worker = worker
        self._shutdown_event = shutdown_event
        self._collect = collect
        self._timesteps = int(timesteps)
        self._max_updates = int(max_updates)
        self._gamma = float(gamma)
        self._gae_lambda = float(gae_lambda)
        self._report = report
        self._after_update = after_update
        self._finish = finish
        self._join_timeout = float(join_timeout)
        self.updates = 0
        self.steps = 0

    def run(self) -> tuple[int, int]:
        """Train until a stop, run ``finish``; returns ``(updates, timesteps)``."""
        self._worker.start()
        try:
            while (self.steps < self._timesteps
                   and not self._shutdown_event.is_set()):
                if 0 < self._max_updates <= self.updates:
                    break
                rollout = self._collect()
                rollout.compute_gae(gamma=self._gamma, gae_lambda=self._gae_lambda)
                self.steps += len(rollout)
                if self._report is not None:
                    self._report(self.updates, rollout, self.steps)
                self._worker.submit(rollout)
                self._worker.wait_idle()
                self.updates += 1
                if self._after_update is not None and self._after_update(self.updates):
                    logger.info("PPOTrainer: stopped by the project at update %d.",
                                self.updates)
                    break
            if self._finish is not None:
                self._finish(self.updates, self.steps)
        finally:
            self._shutdown_event.set()
            self._worker.join(timeout=self._join_timeout)
        return self.updates, self.steps
