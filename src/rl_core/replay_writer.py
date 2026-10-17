"""``ReplayWriter`` — the replay buffer's single push client.

Drains transition batches (anything with ``num_rows``) from the write
queue into ``buffer.push_batch`` — one buffer-lock acquire per actor
step — and publishes the ``transitions_pushed`` counter (single writer;
readers take it lock-free).

Funnelling all pushes through one thread keeps buffer-lock contention
off the processors' step cycle: a processor's ``put`` returns as soon
as the reference lands on the queue, and the slice memcpy into the buffer
happens here, overlapped with everything else (numpy releases the GIL).

On shutdown (event or sentinel) the writer drains whatever is still
queued into the buffer before exiting — queued transitions are valid
data, and the buffer feeds the learner until the run ends.
"""

import logging
import queue
import threading

from ai_runtime.pipeline import SHUTDOWN_SENTINEL

from rl_core.protocols.replay_buffer_protocol import ReplayBufferProtocol
from rl_core.protocols.replay_telemetry_protocol import ReplayTelemetryProtocol

logger = logging.getLogger(__name__)


class ReplayWriter:
    """Writer thread: write queue → ``push_batch``; publishes ``transitions_pushed``."""

    # Queue poll interval; bounds shutdown latency while idle.
    _QUEUE_POLL_TIMEOUT: float = 0.1

    def __init__(
        self,
        replay_buffer: ReplayBufferProtocol,
        write_queue: queue.Queue,
        telemetry: ReplayTelemetryProtocol,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the buffer, the write queue, and the published counter."""
        self._buffer = replay_buffer
        self._write_queue = write_queue
        self._telemetry = telemetry
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        """Writer thread main loop: drain batches into the buffer until shutdown."""
        logger.info("ReplayWriter.run started")
        n_batches = 0
        while not self._shutdown_event.is_set():
            try:
                item = self._write_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if item is SHUTDOWN_SENTINEL:
                break
            if self._push(item):
                n_batches += 1

        # Drain: queued transitions are valid data — push them before exit.
        n_drained = 0
        while True:
            try:
                item = self._write_queue.get_nowait()
            except queue.Empty:
                break
            if item is SHUTDOWN_SENTINEL:
                continue
            if self._push(item):
                n_drained += 1

        logger.info(
            "ReplayWriter.run exiting: %d batches pushed (+%d drained at "
            "shutdown), transitions_pushed=%d",
            n_batches, n_drained, self._telemetry.transitions_pushed,
        )

    def _push(self, batch) -> bool:
        """Push one batch; log and drop it on error rather than killing the thread."""
        try:
            self._buffer.push_batch(batch)
        except Exception as exc:
            logger.exception(
                "ReplayWriter: push_batch failed (rows=%s, error=%s); batch dropped",
                getattr(batch, "num_rows", "?"), exc,
            )
            return False
        self._telemetry.note_transitions(batch.num_rows)
        return True
