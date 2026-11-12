"""``EvalStore`` — ``eval.db``: one dev row per scored checkpoint, the test metrics, the best checkpoint."""

from pathlib import Path
from typing import Optional

from ai_runtime.dev_selection_store import DevSelectionStore


class EvalStore(DevSelectionStore):
    """Dev rows keyed by update count (the ``epoch`` column), test metrics as pairs, one selection criterion."""

    FILENAME = "eval.db"
    SCHEMA_PATH = Path(__file__).with_name("eval_schema.sql")

    def __init__(self, eval_dir: Path, criterion: str) -> None:
        """Open the store under ``eval_dir``; ``criterion`` names the dev column where higher is better."""
        super().__init__(eval_dir)
        self._criterion = criterion

    def add_test(self, metrics: dict) -> None:
        """The ``test_metrics`` pairs of the checkpoint dev chose."""
        self.set_pairs("test_metrics", metrics)

    def test_metrics(self) -> dict:
        """The ``test_metrics`` pairs; empty until the test pass has run."""
        return self.read_pairs("test_metrics")

    def dev_rows(self) -> list[dict]:
        """Every ``dev_metrics`` row, by update count."""
        return self.read("dev_metrics", order_by="epoch")

    def scored(self) -> list[tuple[int, float, float]]:
        """``(update, criterion, bar start)`` of every dev row; the bar starts below any score."""
        return [(int(row["epoch"]), float(row[self._criterion]), float("-inf")) for row in self.dev_rows()]

    def champion(self) -> Optional[tuple[int, str]]:
        """``(update, checkpoint)`` of the best dev row; None before any row."""
        rows = self.dev_rows()
        if not rows:
            return None
        best = max(rows, key=lambda row: float(row[self._criterion]))
        return int(best["epoch"]), str(best["checkpoint"])
