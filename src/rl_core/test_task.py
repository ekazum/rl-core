"""``TestTask`` — scores the best dev checkpoint on test, once, when training has ended."""

from abc import abstractmethod

from rl_core.eval_task import EvalTask


class TestTask(EvalTask):
    """The dev champion on the test split; the project loads it, scores it and writes the test metrics."""

    @abstractmethod
    def run(self) -> dict:
        """Score the best dev checkpoint on test, record it, and return its metrics."""
