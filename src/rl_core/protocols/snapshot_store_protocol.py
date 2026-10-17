"""``SnapshotStoreProtocol`` — where ``CheckpointWriter`` puts snapshots and the latest state."""

from pathlib import Path
from typing import Protocol


class SnapshotStoreProtocol(Protocol):
    """Snapshot files, the latest training state, and retention."""

    def write_snapshot(self, update: int, q_online_state_dict: dict, dims: dict) -> Path:
        """Write one network snapshot; return its path."""

    def write_latest(self, state) -> None:
        """Write the full training state for resume."""

    def delete_snapshot(self, path: Path) -> None:
        """Remove one snapshot file."""
