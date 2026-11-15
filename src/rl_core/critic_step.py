"""``CriticStep`` — what a ``DDPGLearner`` subclass's critic backward pass returns."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CriticStep:
    """The buffer rows of the batch, their new priorities, and the step's diagnostics."""

    # (B,) int64 buffer rows, and (B,) float |TD error| of each, for the priority update.
    indices: np.ndarray
    td_abs: np.ndarray
    # Scalar diagnostics of the step, by name.
    metrics: dict[str, float] = field(default_factory=dict)
