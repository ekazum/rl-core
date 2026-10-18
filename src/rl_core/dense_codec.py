"""``DenseCodec`` — stores each named field of a batch as-is."""

import numpy as np


class DenseCodec:
    """Column per field: ``encode`` reads ``batch.<name>`` and checks its shape."""

    def __init__(self, spec: dict[str, tuple[tuple[int, ...], np.dtype | type]]) -> None:
        """``spec`` is ``name -> (row_shape, dtype)``; each name is a batch attribute."""
        if not spec:
            raise ValueError("spec must name at least one field")
        self._spec = {name: (tuple(shape), np.dtype(dtype))
                      for name, (shape, dtype) in spec.items()}

    def columns(self) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        """The spec, as given."""
        return dict(self._spec)

    def encode(self, batch) -> dict[str, np.ndarray]:
        """``batch.<name>`` cast to the column dtype, shape-checked."""
        n = batch.num_rows
        rows = {}
        for name, (shape, dtype) in self._spec.items():
            arr = np.asarray(getattr(batch, name), dtype=dtype)
            if arr.shape != (n, *shape):
                raise ValueError(f"{name} must be {(n, *shape)}; got {arr.shape}")
            rows[name] = arr
        return rows
