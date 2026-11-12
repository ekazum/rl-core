"""``RowSampler`` — row indices of a table in a fresh random order per pass.

Pass ``k`` is a permutation drawn from a child of the master ``SeedSequence``
with ``spawn_key + (k,)``, so the order of any pass is reproducible and a
resumed run regenerates later passes at the same index.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RowSampler:
    """Every row once per pass, in a per-pass random order; a snapshot restores the position."""

    def __init__(self, n_rows: int, seed_seq: np.random.SeedSequence) -> None:
        """Sample from rows ``0 .. n_rows - 1``."""
        if n_rows < 1:
            raise ValueError(f"RowSampler: n_rows must be at least 1; got {n_rows}")
        self._n_rows = n_rows
        self._seq = seed_seq
        self._pass_counter = 0
        self._cursor = 0
        self._order = self._shuffle_pass(self._pass_counter)

    @property
    def n_rows(self) -> int:
        """Rows in one pass."""
        return self._n_rows

    @property
    def pass_counter(self) -> int:
        """Passes completed before the current one."""
        return self._pass_counter

    def next(self) -> int:
        """The next row of the current pass; a new pass starts when this one is used up."""
        if self._cursor >= self._n_rows:
            self._pass_counter += 1
            self._order = self._shuffle_pass(self._pass_counter)
            self._cursor = 0
            logger.info("RowSampler: starting pass %d (%d rows)", self._pass_counter, self._n_rows)
        row = int(self._order[self._cursor])
        self._cursor += 1
        return row

    def get_snapshot(self) -> dict:
        """Pass counter and cursor; the pass order is regenerated from the seed on load."""
        return {"pass_counter": self._pass_counter, "cursor": self._cursor}

    def load_snapshot(self, snapshot: dict) -> None:
        """Continue from the row after the one the snapshot was taken at."""
        self._pass_counter = int(snapshot["pass_counter"])
        self._cursor = int(snapshot["cursor"])
        self._order = self._shuffle_pass(self._pass_counter)

    def _shuffle_pass(self, pass_counter: int) -> np.ndarray:
        """The row order of pass ``pass_counter``."""
        child = np.random.SeedSequence(entropy=self._seq.entropy, spawn_key=self._seq.spawn_key + (pass_counter,))
        return np.random.default_rng(child).permutation(self._n_rows)
