"""``DQNBatch`` — one training batch on the learner device, as ``DQNLearner``'s ``prepare`` returns it."""

from typing import NamedTuple

import numpy as np
import torch


class DQNBatch(NamedTuple):
    """Dense tensors on the learner device, plus the buffer rows they came from."""

    states: torch.Tensor        # (B, D) float32
    next_states: torch.Tensor   # (B, D) float32
    actions: torch.Tensor       # (B,)   int64
    rewards: torch.Tensor       # (B,)   float32
    dones: torch.Tensor         # (B,)   bool — terminal, not truncated
    is_weights: torch.Tensor    # (B,)   float32
    indices: np.ndarray         # (B,)   int64, for the priority update
    durations: torch.Tensor | None = None   # (B,) float32 τ for an SMDP; None = one step
