"""``SelectTask`` — scores each new checkpoint on dev and records it, so the best one can be chosen."""

from abc import abstractmethod
from pathlib import Path

from rl_core.eval_task import EvalTask


class SelectTask(EvalTask):
    """One checkpoint on the dev split per call; the project loads it, scores it and writes its dev row."""

    @abstractmethod
    def run(self, update: int, checkpoint: Path) -> dict:
        """Score the checkpoint taken at ``update`` on dev, record it, and return its metrics row."""
