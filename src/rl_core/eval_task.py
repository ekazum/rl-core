"""``EvalTask`` — what a training pipeline's manager calls to evaluate a policy."""

from abc import ABC, abstractmethod
from typing import Any


class EvalTask(ABC):
    """Base of the score, select and test tasks; a project fills in ``run``.

    Tasks are independent: none holds another, and the manager calls each.
    """

    # Any: each task names its own arguments.
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict:
        """Run the task once and return its metrics row."""
