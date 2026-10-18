"""``ColumnStore`` — preallocated named columns, one row per stored item.

Row ``i`` of every column is one item; there is no per-row object. Writes
are slice copies, reads are ``take`` into arrays the caller owns, so both
run in numpy with the GIL released. No lock of its own: the owner
serializes writes against reads.
"""

import numpy as np

ColumnSpec = dict[str, tuple[tuple[int, ...], np.dtype | type]]


class ColumnStore:
    """``capacity`` rows of each column in ``spec`` (``name -> (row_shape, dtype)``)."""

    def __init__(self, capacity: int, spec: ColumnSpec) -> None:
        """Allocate the zeroed columns."""
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}")
        if not spec:
            raise ValueError("spec must name at least one column")
        self._capacity = int(capacity)
        self._columns = {
            name: np.zeros((self._capacity, *shape), dtype=dtype)
            for name, (shape, dtype) in spec.items()
        }

    @property
    def capacity(self) -> int:
        """Rows per column."""
        return self._capacity

    @property
    def names(self) -> tuple[str, ...]:
        """Column names, in spec order."""
        return tuple(self._columns)

    def column(self, name: str) -> np.ndarray:
        """The full column array (a live view; read it under the owner's lock)."""
        return self._columns[name]

    def write(self, dst: slice, rows: dict[str, np.ndarray], src: slice) -> None:
        """Copy ``rows[name][src]`` into slots ``dst`` of every column."""
        for name, col in self._columns.items():
            col[dst] = rows[name][src]

    def take(self, indices: np.ndarray, out: dict[str, np.ndarray]) -> None:
        """Gather rows ``indices`` of the columns named in ``out`` into its arrays.

        ``mode="clip"`` skips the bounds check; indices must be in ``[0, capacity)``.
        """
        for name, dst in out.items():
            self._columns[name].take(indices, axis=0, out=dst, mode="clip")
