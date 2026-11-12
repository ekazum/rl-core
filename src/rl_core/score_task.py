"""``ScoreTask`` — scores the current policy on one split."""

from abc import abstractmethod

from rl_core.eval_task import EvalTask


class ScoreTask(EvalTask):
    """One policy on one split; the project produces the outputs, the rewards and the record."""

    @abstractmethod
    def run(self, split: str) -> dict:
        """Score the current policy on ``split`` and return its metrics row."""
