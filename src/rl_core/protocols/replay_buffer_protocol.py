"""``ReplayBufferProtocol`` — what the replay threads need from a buffer.

``ReplayWriter``, ``ReplayReader``, ``PriorityUpdater`` and ``StatsProcessor``
call these methods only; the buffer itself is the project's.
"""

from typing import Protocol

import numpy as np


class ReplayBufferProtocol(Protocol):
    """A prioritized replay buffer, as its client threads use it."""

    def push_batch(self, batch) -> None:
        """Store one step's transitions; ``batch.num_rows`` of them."""

    def draw_block(self, batch_size: int,
                   n_batches: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """``n_batches`` prioritized ``(indices, is_weights)`` draws."""

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """New priorities for the given rows from their TD errors."""

    def get_priority_stats(self) -> dict:
        """Priority spread, β and occupancy; ``size`` among them."""

    def __len__(self) -> int:
        """Transitions stored."""
