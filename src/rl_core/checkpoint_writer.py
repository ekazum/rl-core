"""``CheckpointWriter`` — the checkpoint I/O thread.

Keeps the filesystem off the learner, the way ``PriorityUpdater`` keeps
the priority write off it. The learner clones ``q_online`` under
``network_lock`` — only it can copy consistently — and hands the clone
over; everything after that happens here: the snapshot file, the
retention drop, and the latest-checkpoint write. A bounded write queue is
backpressure for a slow filesystem and a cap on queued memory.

The ``CheckpointEvent`` that makes ``StatsProcessor`` write its
buffer-stats row is emitted **after** the snapshot lands, so no row can
name a checkpoint that failed to write.
"""

import logging
import queue
import threading
from pathlib import Path

from ai_runtime.pipeline import SHUTDOWN_SENTINEL

from rl_core.checkpoint_state import CheckpointState
from rl_core.events.checkpoint_event import CheckpointEvent
from rl_core.events.checkpoint_request import CheckpointRequest
from rl_core.events.latest_request import LatestRequest
from rl_core.protocols.snapshot_store_protocol import SnapshotStoreProtocol

logger = logging.getLogger(__name__)


class CheckpointWriter:
    """Drains write requests: snapshot + retention, and latest."""

    # Queue poll interval; bounds how fast shutdown is observed.
    _QUEUE_POLL_TIMEOUT: float = 0.5

    def __init__(
        self,
        write_queue: "queue.Queue",
        stats_q: "queue.Queue",
        checkpoint_manager: SnapshotStoreProtocol,
        shutdown_event: threading.Event,
        keep_all: bool,
    ) -> None:
        """Bind the queues, the checkpoint owner, and the retention policy."""
        self._queue = write_queue
        self._stats_q = stats_q
        self._checkpoint_manager = checkpoint_manager
        self._shutdown_event = shutdown_event
        self._keep_all = bool(keep_all)
        # The surviving snapshot, when only the newest is kept.
        self._kept: tuple[int, Path] | None = None
        self._written: int = 0

    def run(self) -> None:
        """Thread main loop: drain until the sentinel, then exit.

        Requests queued before shutdown are drained first — the learner's
        final checkpoint must land before the process exits.
        """
        logger.info("CheckpointWriter.run started")
        while True:
            try:
                item = self._queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                if self._shutdown_event.is_set() and self._queue.empty():
                    break
                continue
            if item is SHUTDOWN_SENTINEL:
                break
            try:
                self._dispatch(item)
            except Exception as exc:                      # noqa: BLE001
                logger.exception(
                    "CheckpointWriter: write failed (%s); continuing", exc)
        logger.info("CheckpointWriter.run exiting: %d checkpoint(s) written",
                    self._written)

    def _dispatch(self, item) -> None:
        """Apply one request; unknown types are logged, not fatal."""
        if isinstance(item, CheckpointRequest):
            self._write_snapshot(item)
        elif isinstance(item, LatestRequest):
            if not isinstance(item.state, CheckpointState):
                raise RuntimeError(f"LatestRequest.state is {type(item.state).__name__}, "
                                   "not CheckpointState")
            self._checkpoint_manager.write_latest(item.state)
        else:
            logger.warning("CheckpointWriter: unknown request %r", type(item))

    def _write_snapshot(self, request: CheckpointRequest) -> None:
        """Write the snapshot, apply retention, tick the stats row."""
        path = self._checkpoint_manager.write_snapshot(
            request.update, request.state_dict, request.dims,
        )
        if not self._keep_all and self._kept is not None:
            self._checkpoint_manager.delete_snapshot(self._kept[1])
        self._kept = (request.update, path)
        self._written += 1
        # Only now: the row must not name a checkpoint that failed to land.
        self._stats_q.put(CheckpointEvent(update_at=request.update))
