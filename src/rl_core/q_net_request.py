"""``QNetRequest`` — one batch of states awaiting their Q-values from ``QNetworkWorker``."""

from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np


@dataclass
class QNetRequest:
    """One q-value request: ``(L, D)`` live-state matrix awaiting ``(L, A)`` q-values."""

    vectors: np.ndarray
    response: Future
