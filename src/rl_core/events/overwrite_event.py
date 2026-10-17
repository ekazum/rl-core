"""``OverwriteEvent`` — slot ranges a buffer push overwrote; best-effort like ``DrawEvent``."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OverwriteEvent:
    """Slot range(s) ``push_batch`` overwrote — their draw counts reset to 0.

    ``ranges`` is a tuple of ``(lo, hi)`` half-open slot ranges (one normally,
    two on a wraparound).
    """

    ranges: tuple[tuple[int, int], ...]
