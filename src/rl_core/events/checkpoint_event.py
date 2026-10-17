"""``CheckpointEvent`` — a checkpoint landed; reliable, never dropped."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointEvent:
    """A checkpoint was written — the processor writes the buffer-stats row.

    ``update_at`` is the learner update count the snapshot was taken at, and
    the key the row is stored under. Emitted by the ``CheckpointWriter``
    once the file is on disk.
    """

    update_at: int
