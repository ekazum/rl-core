"""``ReplayReader`` — the replay buffer's single draw client.

Keeps the prefetch queue stocked with ``(indices, is_weights)`` draws so
the priority-weighted draw never lands on the learner's critical path; the
data gathers happen downstream, concurrently under the buffer's shared
lock. The reader never touches transition data. Owns the
``min_buffer_size`` warmup gate: the queue stays empty until the buffer
fills, which is how the learner's start is gated.

Prefetched draws use priorities up to ``block_batches`` draws old, plus the
prefetch queue's depth and in-flight gathers — the standard trade of
prefetched prioritized replay.
"""

import logging
import queue
import threading

from ai_runtime.thread_handle import ThreadHandle

from rl_core.protocols.replay_buffer_protocol import ReplayBufferProtocol

logger = logging.getLogger(__name__)


class ReplayReader:
    """Reader thread: prefetch ``batch_size`` draws into the bounded prefetch queue."""

    # Buffer-fill poll interval during warmup.
    _WARMUP_POLL_SECONDS: float = 0.1
    # Bounded-put retry interval; bounds shutdown latency while the queue is full.
    _QUEUE_PUT_TIMEOUT: float = 0.5
    # Back-off after a failed sample before retrying.
    _ERROR_BACKOFF_SECONDS: float = 0.5

    def __init__(
        self,
        replay_buffer: ReplayBufferProtocol,
        prefetch_queue: queue.Queue,
        batch_size: int,
        min_buffer_size: int,
        block_batches: int,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the buffer, the prefetch queue and the draw sizes."""
        self._buffer = replay_buffer
        self._prefetch_queue = prefetch_queue
        self._batch_size = int(batch_size)
        self._block_batches = int(block_batches)
        self._shutdown_event = shutdown_event
        # Sampling needs at least batch_size rows regardless of the
        # configured warmup floor.
        self._warmup_size = max(int(min_buffer_size), self._batch_size)

    def run(self) -> None:
        """Reader thread main loop: warmup gate, then the prefetch loop."""
        logger.info(
            "ReplayReader.run waiting for buffer to reach %d transitions ...",
            self._warmup_size,
        )
        if not self._wait_for_warmup():
            logger.info("ReplayReader.run shut down during warmup")
            return
        logger.info(
            "ReplayReader.run warmup done; buffer has %d transitions. Prefetching...",
            len(self._buffer),
        )
        n_sampled = 0
        block_batches = self._block_batches
        while not self._shutdown_event.is_set():
            # --- DEBUG wait-localization markers (read deltas from timestamps) ---
            # start->done = buffer read-lock acquire + draw (feed-bound if large);
            # done->queued = blocking on a full prefetch queue (learner-bound if large).
            logger.debug("reader: draw_block start")
            try:
                items = self._buffer.draw_block(
                    self._batch_size, block_batches,
                )
            except Exception as exc:
                logger.exception(
                    "ReplayReader: draw_block failed (%s); backing off", exc,
                )
                ThreadHandle.interruptible_sleep(
                    self._shutdown_event, self._ERROR_BACKOFF_SECONDS,
                )
                continue
            logger.debug("reader: draw_block done (%d items)", len(items))
            aborted = False
            for item in items:
                if not self._put(item):
                    aborted = True  # shutdown observed while the queue was full
                    break
                n_sampled += 1
            if aborted:
                break
            logger.debug("reader: items queued")
        logger.info("ReplayReader.run exiting: %d batches prefetched", n_sampled)

    def _wait_for_warmup(self) -> bool:
        """Block until the buffer holds enough rows; False on shutdown."""
        while not self._shutdown_event.is_set():
            if len(self._buffer) >= self._warmup_size:
                return True
            ThreadHandle.interruptible_sleep(
                self._shutdown_event, self._WARMUP_POLL_SECONDS,
            )
        return False

    def _put(self, item) -> bool:
        """Bounded put that stays shutdown-responsive; False on shutdown.

        A full prefetch queue is the reader's backpressure — the learner
        pipeline is still consuming earlier batches.
        """
        while not self._shutdown_event.is_set():
            try:
                self._prefetch_queue.put(item, timeout=self._QUEUE_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False
