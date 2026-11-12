"""``EvalConfig`` — the evaluation settings every project shares; a project subclass adds its own."""

from typing import Any

from ai_runtime.yaml_config import YamlConfig


class EvalConfig(YamlConfig):
    """Batch size and row limit of an evaluation pass."""

    # Rows scored per batch.
    batch_size: int
    # Rows scored per split; 0 scores every row.
    limit: int = 0

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        if int(raw.get("batch_size", 0)) < 1:
            raise ValueError("EvalConfig: batch_size must be at least 1")
        if int(raw.get("limit", 0)) < 0:
            raise ValueError("EvalConfig: limit must be 0 or more")
