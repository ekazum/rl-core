"""``TargetRequest`` — one batch awaiting its target-network values from ``QTargetWorker``."""

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any


@dataclass
class TargetRequest:
    """``inputs`` for the worker's ``decode`` (default: the ``(B, D)`` tensor itself) → ``(B, A)``."""

    inputs: Any
    response: Future
