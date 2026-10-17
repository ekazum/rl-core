"""``PPORolloutBatch`` — one actor batch's transitions, with GAE.

Transitions are appended step-major, so a given episode's rows already sit in
increasing step order and a stable sort by ``rows`` groups them. Advantages
follow the standard recursion, A_t = d_t + gamma*lam*A_{t+1} with
d_t = r_t + gamma*V(s_{t+1}) - V(s_t), bootstrapping nothing past a done.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PPORolloutBatch:
    """Flat transitions of one actor batch, plus what the update needs."""

    observations: np.ndarray                 # (N, obs_dim) float32
    actions: np.ndarray                      # (N,)         int64
    log_probs: np.ndarray                    # (N,)         float32
    values: np.ndarray                       # (N,)         float32
    rewards: np.ndarray                      # (N,)         float32
    dones: np.ndarray                        # (N,)         bool
    invalid: np.ndarray                      # (N, A)       bool
    rows: np.ndarray                         # (N,)         int64 — episode id
    advantages: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    returns: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    @classmethod
    def concatenate(cls, parts: list["PPORolloutBatch"]) -> "PPORolloutBatch":
        """Join per-step pieces into one batch, keeping append order."""
        if not parts:
            raise ValueError("no rollout pieces to concatenate")
        return cls(
            observations=np.concatenate([p.observations for p in parts]),
            actions=np.concatenate([p.actions for p in parts]),
            log_probs=np.concatenate([p.log_probs for p in parts]),
            values=np.concatenate([p.values for p in parts]),
            rewards=np.concatenate([p.rewards for p in parts]),
            dones=np.concatenate([p.dones for p in parts]),
            invalid=np.concatenate([p.invalid for p in parts]),
            rows=np.concatenate([p.rows for p in parts]),
        )

    def compute_gae(self, gamma: float = 1.0,
                    gae_lambda: float = 0.95) -> None:
        """Fill ``advantages`` and ``returns`` in place, per episode.

        A row that ran out of steps without predicting is treated as terminal:
        nothing remains to be earned past the cap, so no value bootstraps
        across it.
        """
        order = np.argsort(self.rows, kind="stable")                # (N,)
        advantages = np.zeros(len(self), dtype=np.float32)
        carry = 0.0
        next_value = 0.0
        previous_row = -1
        for position in order[::-1]:
            row = int(self.rows[position])
            if row != previous_row:                                 # episode's last step
                carry = 0.0
                next_value = 0.0
                previous_row = row
            not_done = 0.0 if self.dones[position] else 1.0
            delta = (float(self.rewards[position])
                     + gamma * next_value * not_done
                     - float(self.values[position]))
            carry = delta + gamma * gae_lambda * not_done * carry
            advantages[position] = carry
            next_value = float(self.values[position])
        self.advantages = advantages
        self.returns = (advantages + self.values).astype(np.float32)

    def stats(self) -> dict:
        """One row of the numbers the log reports for this rollout."""
        return {
            "transitions": len(self),
            "episodes": int(np.unique(self.rows).size),
            "reward_mean": float(self.rewards.mean()),
            "reward_min": float(self.rewards.min()),
            "reward_max": float(self.rewards.max()),
            "value_mean": float(self.values.mean()),
            "value_min": float(self.values.min()),
            "value_max": float(self.values.max()),
            "advantage_mean": float(self.advantages.mean()) if self.advantages.size else 0.0,
            "advantage_std": float(self.advantages.std()) if self.advantages.size else 0.0,
            "return_mean": float(self.returns.mean()) if self.returns.size else 0.0,
            "return_min": float(self.returns.min()) if self.returns.size else 0.0,
            "return_max": float(self.returns.max()) if self.returns.size else 0.0,
        }
