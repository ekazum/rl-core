"""``SyncRequest`` — a ``q_online`` snapshot for ``QTargetWorker`` to load into ``q_target``."""

from dataclasses import dataclass


@dataclass
class SyncRequest:
    """A ``q_online.state_dict()`` snapshot to load into ``q_target``."""

    state_dict: dict
