"""``CheckpointState`` — schema for the full persisted training state.

Held in memory only long enough to round-trip through the snapshot
store's ``write_latest`` / ``read_latest``; not a long-lived runtime object.

Each component owns its own ``get_state`` / ``get_snapshot`` format;
``CheckpointState`` is a pass-through schema that collects them:

* ``q_online_state_dict`` / ``q_target_state_dict`` — the two networks
* ``learner_state``     — optimizer state, the update count and the
  learner's schedule cursors
* ``validator_state``   — validation results (best dev score, patience)
* ``episode_count``     — the shared episode counter
* ``episode_sampler_snapshot`` — where the episode sampler stands
* ``config_hash``       — the run's config, checked on resume

The replay buffer is intentionally not part of the checkpoint: it re-warms
from the actors' continued production on resume, so snapshotting it
(often GBs) would only add a slow write and a long exit stall.

This keeps each component's serialisation format private to that
component — ``CheckpointState`` is a pass-through schema for
orchestration only.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Schema for the full checkpoint state written by ``CheckpointManager.write_latest``."""

    q_online_state_dict: dict
    q_target_state_dict: dict
    learner_state: dict
    episode_sampler_snapshot: dict
    episode_count: int
    config_hash: str
    validator_state: dict = field(default_factory=dict)
    transitions_pushed: int = 0

    def to_dict(self) -> dict:
        """Flatten to a dict suitable for ``torch.save``.

        Shallow dict — nested ``state_dict`` / ``snapshot`` payloads
        are passed through by reference (not deep-copied) so we don't
        duplicate large tensors or arrays during serialisation.
        """
        return {
            "q_online_state_dict": self.q_online_state_dict,
            "q_target_state_dict": self.q_target_state_dict,
            "learner_state": self.learner_state,
            "validator_state": self.validator_state,
            "episode_sampler_snapshot": self.episode_sampler_snapshot,
            "episode_count": int(self.episode_count),
            "config_hash": str(self.config_hash),
            "transitions_pushed": int(self.transitions_pushed),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CheckpointState":
        """Reconstruct a ``CheckpointState`` from a dict produced by ``to_dict``."""
        return cls(
            q_online_state_dict=d["q_online_state_dict"],
            q_target_state_dict=d["q_target_state_dict"],
            learner_state=d["learner_state"],
            validator_state=d.get("validator_state", {}),
            episode_sampler_snapshot=d["episode_sampler_snapshot"],
            episode_count=int(d["episode_count"]),
            config_hash=str(d["config_hash"]),
            transitions_pushed=int(d.get("transitions_pushed", 0)),
        )
