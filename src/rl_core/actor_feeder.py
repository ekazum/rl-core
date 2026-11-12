"""``ActorFeeder`` — the feeder thread: row ids from a ``RowSampler``, cut into actor batches.

Each batch is a ``list[int]`` of ``batch_size`` rows put on the batch queue; the
actor processors load the data. ``max_rows > 0`` stops dispatching once that many
rows are out; 0 dispatches until shutdown. ``push_per_update_ratio > 0`` waits while
``transitions_pushed > ratio * update_count``, and is inactive before the first update.
"""

import logging
import queue
import threading

from rl_core.protocols.feeder_telemetry_protocol import FeederTelemetryProtocol
from rl_core.row_sampler import RowSampler

logger = logging.getLogger(__name__)


class ActorFeeder:
    """Cuts the sampler's order into id batches, paces them, and puts them on the batch queue."""

    # Bounded-put retry interval; also bounds shutdown latency while blocked.
    _QUEUE_PUT_TIMEOUT: float = 0.5
    # Throttle re-check interval.
    _THROTTLE_POLL_SECONDS: float = 0.05

    def __init__(
        self,
        sampler: RowSampler,
        batch_queue: queue.Queue,
        telemetry: FeederTelemetryProtocol,
        batch_size: int,
        max_rows: int,
        push_per_update_ratio: float,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the sampler, the batch queue and the counters the throttle reads."""
        if batch_size < 1:
            raise ValueError(f"ActorFeeder: batch_size must be at least 1; got {batch_size}")
        if max_rows < 0:
            raise ValueError(f"ActorFeeder: max_rows must be 0 or more; got {max_rows}")
        self._sampler = sampler
        self._batch_queue = batch_queue
        self._telemetry = telemetry
        self._batch_size = batch_size
        self._max_rows = max_rows
        self._ratio = push_per_update_ratio
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        """Dispatch batches until the row cap or shutdown; a failure sets the shutdown event."""
        logger.info("ActorFeeder: batch_size=%d, max_rows=%s, push_per_update_ratio=%s", self._batch_size,
                    self._max_rows or "uncapped", self._ratio if self._ratio > 0 else "off")
        dispatched = 0
        try:
            while not self._shutdown_event.is_set():
                if 0 < self._max_rows <= dispatched:
                    logger.info("ActorFeeder: max_rows=%d dispatched; production complete", self._max_rows)
                    break
                if not self._throttle():
                    break
                rows = [self._sampler.next() for _ in range(self._batch_size)]
                if not self._dispatch(rows):
                    break
                dispatched += len(rows)
        except Exception:
            self._shutdown_event.set()
            raise
        logger.info("ActorFeeder: exiting, %d rows dispatched", dispatched)

    def _throttle(self) -> bool:
        """Wait while production runs ahead of training; False on shutdown."""
        if self._ratio <= 0:
            return True
        while (self._telemetry.update_count > 0
               and self._telemetry.transitions_pushed > self._ratio * self._telemetry.update_count):
            if self._shutdown_event.wait(self._THROTTLE_POLL_SECONDS):
                return False
        return True

    def _dispatch(self, rows: list[int]) -> bool:
        """Bounded put that observes shutdown; False on shutdown."""
        while not self._shutdown_event.is_set():
            try:
                self._batch_queue.put(rows, timeout=self._QUEUE_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False
