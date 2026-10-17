"""``SumTree`` — vectorized binary segment tree over per-slot priorities.

A replay buffer indexes its
``_priorities`` with this tree so prioritized sampling and priority updates are
O(log n) instead of an O(n) ``cumsum`` rebuilt under the buffer lock on every
draw. The tree stores only priorities (the root holds their total); the buffer
owns the transition data and gathers it at the returned indices.

Layout: leaves are padded to the next power of two ``cap``; the tree is a flat
``float64`` array of length ``2 * cap`` with internal node ``k`` holding the sum
of children ``2k`` and ``2k+1``, leaf ``i`` at ``tree[cap + i]``, and the total
at ``tree[1]``. Leaves in ``[capacity, cap)`` (the pad) stay 0 and are therefore
unreachable by sampling.

All operations are vectorized (batched numpy) — a per-element Python descent
would be slower than numpy's C ``cumsum``. ``sample`` reproduces
``np.searchsorted(np.cumsum(leaves), u, side="right")`` exactly.

Thread-safety: none of its own — the buffer calls it only under its exclusive
``write_lock`` (single source of mutation), so the tree needs no internal lock.
"""

import numpy as np


class SumTree:
    """Vectorized sum-tree (segment tree of priority sums) over ``capacity`` leaves."""

    def __init__(self, capacity: int) -> None:
        """Allocate a zeroed tree sized to the next power of two ≥ ``capacity``."""
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}")
        self._capacity = int(capacity)
        cap = 1
        while cap < self._capacity:
            cap <<= 1
        self._cap = cap
        self._tree = np.zeros(2 * cap, dtype=np.float64)

    @property
    def total(self) -> float:
        """Sum of all leaf priorities (the root). O(1)."""
        return float(self._tree[1])

    def build(self, leaves: np.ndarray) -> None:
        """Bulk (re)build the tree from a priority array of length ≤ ``capacity``.

        One pass per level (vectorized) — O(cap). Used only at construction-time
        equivalents / explicit rebuilds, not on the per-draw hot path.
        """
        leaves = np.asarray(leaves, dtype=np.float64).ravel()
        if leaves.shape[0] > self._capacity:
            raise ValueError(
                f"leaves length {leaves.shape[0]} exceeds capacity {self._capacity}"
            )
        self._tree[:] = 0.0
        self._tree[self._cap : self._cap + leaves.shape[0]] = leaves
        size = self._cap
        while size > 1:
            size //= 2
            nodes = np.arange(size, 2 * size)
            self._tree[nodes] = self._tree[2 * nodes] + self._tree[2 * nodes + 1]

    def update(self, idx: np.ndarray, vals: np.ndarray) -> None:
        """Set leaves ``idx`` to ``vals`` and recompute their ancestors.

        Duplicate indices resolve last-write-wins (numpy fancy assignment),
        matching ``_priorities[idx] = vals``. Ancestors are recomputed
        level-by-level over the unique affected nodes — O(B·log n).
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return
        vals = np.asarray(vals, dtype=np.float64).ravel()
        self._tree[self._cap + idx] = vals          # last-write-wins for dup idx
        nodes = self._cap + idx
        while True:
            nodes = np.unique(nodes // 2)            # parent level
            self._tree[nodes] = self._tree[2 * nodes] + self._tree[2 * nodes + 1]
            if nodes[0] == 1:                        # recomputed the root
                break

    def sample(self, u: np.ndarray) -> np.ndarray:
        """Map cumulative-weight queries ``u`` (each in ``[0, total)``) to leaf indices.

        Batched root→leaf descent: at each level go right when ``u`` reaches the
        left child's sum, else go left — the inverse-CDF lookup. Reproduces
        ``np.searchsorted(np.cumsum(leaves), u, side="right")``. O(B·log n).
        """
        u = np.asarray(u, dtype=np.float64).ravel().copy()
        pos = np.ones(u.shape[0], dtype=np.int64)
        if u.shape[0] == 0:
            return pos - self._cap
        while pos[0] < self._cap:                    # all positions descend in lockstep
            left = 2 * pos
            left_sum = self._tree[left]
            go_right = u >= left_sum
            u -= np.where(go_right, left_sum, 0.0)
            pos = left + go_right.astype(np.int64)
        return pos - self._cap
