"""``LatestRequest`` — the full training state for ``CheckpointWriter`` to persist."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LatestRequest:
    """A full ``CheckpointState`` to persist as the latest checkpoint."""

    state: object
