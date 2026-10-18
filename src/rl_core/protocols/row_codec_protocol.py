"""``RowCodecProtocol`` — how ``PrioritizedReplayBuffer`` stores a project's transitions.

The codec names the stored columns and turns one pushed batch into one array
per column. ``encode`` runs before the buffer takes its lock, so packing work
there costs the other buffer clients nothing.
"""

from typing import Protocol

import numpy as np


class RowCodecProtocol(Protocol):
    """Column layout and batch encoder for a replay buffer."""

    def columns(self) -> dict[str, tuple[tuple[int, ...], np.dtype | type]]:
        """``name -> (row_shape, dtype)`` of every stored column."""

    def encode(self, batch) -> dict[str, np.ndarray]:
        """One ``(batch.num_rows, *row_shape)`` array per column; raise on a bad batch."""
