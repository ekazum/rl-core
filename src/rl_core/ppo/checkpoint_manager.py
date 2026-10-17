"""``PPOCheckpointManager`` — the policy's two files, plus per-update snapshots.

``latest.pt`` (model + optimizer + update + meta) for resuming, and ``best.pt``
(model + meta), written whenever the dev criterion improves. The network's
constructor arguments ride in every payload so a reader rebuilds it without
the training config.
"""

import logging
from pathlib import Path

import torch
from ai_runtime.checkpoints import LatestBestCheckpointManager

logger = logging.getLogger(__name__)


class PPOCheckpointManager(LatestBestCheckpointManager):
    """``latest.pt`` for resume, ``best.pt`` for the champion."""

    STEP_KEY = "update"
    # One model-only file per saved update; zero-padded so name order is
    # update order.
    SNAPSHOT = "u_{update:06d}.pt"

    def save_snapshot(self, model: torch.nn.Module, update: int,
                      meta: dict) -> Path:
        """Write the policy at ``update`` to its own file, kept for offline scoring."""
        path = self.checkpoint_dir / self.SNAPSHOT.format(update=int(update))
        self.atomic_save({
            "model_state_dict": model.state_dict(),
            "update": int(update),
            "meta": self._meta(model, meta),
        }, path)
        return path

    def list_snapshots(self) -> list[Path]:
        """Every saved snapshot, in update order."""
        return self.snapshots("u_*.pt")

    def clear_snapshots(self) -> int:
        """Delete the snapshots once the champion is tested; the two files stay."""
        paths = self.list_snapshots()
        for path in paths:
            path.unlink()
        logger.info("%s: removed %d snapshots, kept %s and %s",
                    type(self).__name__, len(paths), self.BEST, self.LATEST)
        return len(paths)
