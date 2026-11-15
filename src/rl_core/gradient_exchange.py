"""``GradientExchange`` — gradient averaging between learner threads, each with its own critic copy and device.

Each learner calls ``average(rank)`` after its backward pass. Once all have arrived, rank 0 averages the gradients
parameter by parameter into its own critic and copies the mean into every other copy; then all go on to the same
optimizer step. Every ``check_every`` calls rank 0 also compares the copies' weight sums and, on a gap, copies its
weights into the others. Waits poll ``shutdown_event``; ``average`` returns False once it is set.
"""

import logging
import threading

import torch
from torch import nn

logger = logging.getLogger(__name__)


class GradientExchange:
    """In-process all-reduce of gradients between one learner thread per critic copy."""

    # Seconds between two checks of the shutdown event while waiting for the other learners.
    _POLL_S: float = 0.1

    def __init__(self, critics: list[nn.Module], shutdown_event: threading.Event, check_every: int = 0) -> None:
        """One critic copy per learner, ``critics[r]`` for rank ``r``; ``check_every`` of 0 turns the weight check off."""
        if len(critics) < 2:
            raise ValueError(f"GradientExchange needs at least two critics; got {len(critics)}")
        self._params = [list(critic.parameters()) for critic in critics]
        if len({len(params) for params in self._params}) != 1:
            raise ValueError("GradientExchange: the critics have different parameter counts")
        self._shutdown_event = shutdown_event
        self._check_every = int(check_every)
        self._parties = len(critics)
        self._condition = threading.Condition()
        self._arrived = 0
        self._generation = 0
        self._calls = 0

    def average(self, rank: int) -> bool:
        """Average the gradients of all copies into each; False if shutdown was set while waiting."""
        self._synchronize(rank)
        if not self._rendezvous():
            return False
        if rank == 0:
            self._reduce()
            self._calls += 1
            if self._check_every > 0 and self._calls % self._check_every == 0:
                self._check_weights()
            for r in range(self._parties):
                self._synchronize(r)
        return self._rendezvous()

    def _synchronize(self, rank: int) -> None:
        """Wait for the queued kernels on rank ``rank``'s device."""
        device = self._params[rank][0].device
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _rendezvous(self) -> bool:
        """Block until every learner has arrived; False if shutdown was set first."""
        with self._condition:
            generation = self._generation
            self._arrived += 1
            if self._arrived == self._parties:
                self._arrived = 0
                self._generation += 1
                self._condition.notify_all()
                return True
            while self._generation == generation:
                if self._shutdown_event.is_set():
                    return False
                self._condition.wait(self._POLL_S)
            return True

    def _reduce(self) -> None:
        """g ← mean of the copies' gradients, written into every copy; one parameter on the move at a time."""
        with torch.no_grad():
            for group in zip(*self._params):
                main = group[0].grad
                if main is None:
                    continue
                for other in group[1:]:
                    if other.grad is not None:
                        main.add_(other.grad.to(main.device, non_blocking=True))
                main.div_(self._parties)
                for other in group[1:]:
                    if other.grad is not None:
                        other.grad.copy_(main, non_blocking=True)

    def _check_weights(self) -> None:
        """Compare each copy's weight sum with rank 0's; on a gap, copy rank 0's weights into that copy."""
        with torch.no_grad():
            sums = [float(torch.stack([p.detach().double().sum() for p in params]).sum()) for params in self._params]
            for rank in range(1, self._parties):
                gap = abs(sums[rank] - sums[0])
                if gap == 0.0:
                    continue
                logger.warning("GradientExchange: copy %d weight sum differs from copy 0 by %.6g; copying copy 0 "
                               "into it", rank, gap)
                for main, other in zip(self._params[0], self._params[rank]):
                    other.copy_(main)
            logger.info("GradientExchange: weight check after %d exchanges, sums %s", self._calls,
                        ", ".join(f"{s:.6f}" for s in sums))
