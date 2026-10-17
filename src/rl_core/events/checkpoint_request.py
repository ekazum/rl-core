"""``CheckpointRequest`` — a network snapshot for ``CheckpointWriter`` to write."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointRequest:
    """A snapshot to write — the learner's clone, handed to the writer.

    ``state_dict`` is a CPU copy taken under the network lock, so the
    writer touches nothing the learner is still mutating.
    """

    update: int
    state_dict: dict
    dims: dict
