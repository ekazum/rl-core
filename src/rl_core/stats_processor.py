"""``StatsProcessor`` — the CPU stats thread.

Sole owner of the replay draw
tally (``_sample_counts`` / ``_total_draws``), maintained by draining
best-effort ``DrawEvent`` / ``OverwriteEvent``s the buffer emits — so
``draw_block`` stays a pure reader (no telemetry race). On each
``CheckpointEvent`` it snapshots the buffer's ``get_priority_stats()``,
merges its own replay-concentration stats, and is the **sole writer** of the
buffer-stats rows, keyed by the learner update count the snapshot was
taken at.
"""

import logging
import queue
import threading

import numpy as np
from ai_runtime.pipeline import SHUTDOWN_SENTINEL

from rl_core.events.checkpoint_event import CheckpointEvent
from rl_core.events.draw_event import DrawEvent
from rl_core.events.overwrite_event import OverwriteEvent
from rl_core.protocols.replay_buffer_protocol import ReplayBufferProtocol
from rl_core.protocols.replay_telemetry_protocol import ReplayTelemetryProtocol

logger = logging.getLogger(__name__)


class StatsProcessor:
    """Owns the replay draw tally; writes the buffer-stats rows."""

    # Queue poll interval; bounds how fast shutdown is observed while idle.
    _QUEUE_POLL_TIMEOUT: float = 0.5

    def __init__(
        self,
        stats_q: "queue.Queue",
        replay_buffer: ReplayBufferProtocol,
        telemetry: ReplayTelemetryProtocol,
        capacity: int,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the event queue, buffer (for priority snapshots), and telemetry."""
        self._stats_q = stats_q
        self._replay_buffer = replay_buffer
        self._telemetry = telemetry
        self._shutdown_event = shutdown_event
        # Sole owner of the draw tally (mirrors the buffer's slots via events).
        self._sample_counts = np.zeros(capacity, dtype=np.int64)
        self._total_draws = 0

    def run(self) -> None:
        """Thread main loop: drain stats events; write rows on each result."""
        logger.info("StatsProcessor.run started")
        while not self._shutdown_event.is_set():
            try:
                event = self._stats_q.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if event is SHUTDOWN_SENTINEL:
                break
            try:
                self._dispatch(event)
            except Exception as exc:
                logger.exception(
                    "StatsProcessor: event failed (%s); continuing", exc,
                )
        logger.info(
            "StatsProcessor.run exiting: %d total draws tallied",
            self._total_draws,
        )

    def _dispatch(self, event) -> None:
        """Apply one event to the tally, or write a row on a CheckpointEvent."""
        if isinstance(event, DrawEvent):
            np.add.at(self._sample_counts, event.indices, 1)
            self._total_draws += int(event.indices.size)
        elif isinstance(event, OverwriteEvent):
            for lo, hi in event.ranges:
                self._sample_counts[lo:hi] = 0
        elif isinstance(event, CheckpointEvent):
            self._write_stats(event.update_at)
        else:
            logger.warning("StatsProcessor: unknown event %r", type(event))

    def _write_stats(self, update_at: int) -> None:
        """Write the merged buffer-stats row, keyed by update count."""
        prio = self._replay_buffer.get_priority_stats()
        replay = self._replay_stats(int(prio["size"]))
        self._telemetry.log_buffer_stats(update_at, {**prio, **replay})
        logger.info(
            "ReplayBuffer @ update %d: size=%d β=%.3f prio[max/mean=%.1f H=%.2f] "
            "replay[mean=%.1f max=%d never=%.1f%% top1%%=%.1f%%]",
            update_at, prio["size"], prio["beta_is"],
            prio["prio_max_over_mean"], prio["prio_entropy"],
            replay["replay_mean"], replay["replay_max"],
            100.0 * replay["replay_never_frac"],
            100.0 * replay["replay_top1pct_share"],
        )

    def _replay_stats(self, n: int) -> dict:
        """Replay-concentration stats over the occupied slots (the VP's tally)."""
        if n == 0:
            return {
                "replay_mean": 0.0, "replay_max": 0,
                "replay_never_frac": 0.0, "replay_top1pct_share": 0.0,
                "total_draws": self._total_draws,
            }
        counts = self._sample_counts[:n]
        draws_total = int(counts.sum())
        top_k = max(1, n // 100)
        top_share = (
            float(np.sort(counts)[-top_k:].sum()) / draws_total
            if draws_total > 0 else 0.0
        )
        return {
            "replay_mean": float(counts.mean()),
            "replay_max": int(counts.max()),
            "replay_never_frac": float((counts == 0).mean()),
            "replay_top1pct_share": top_share,
            "total_draws": self._total_draws,
        }
