"""``PriorityUpdater`` — the replay buffer's single priority-update client.

Applies the learner's TD
errors to the buffer without putting the buffer lock on the update hot
path: the learner drops ``(indices, |td|)`` on the priority queue and
moves on; this thread applies them.

Staleness note: an index sampled by the reader can be overwritten by
the writer before its priority update lands, in which case the fresh
transition briefly carries a TD-based priority instead of max-priority.
The race window is bounded by the priority-queue depth; priorities are
sampling heuristics, and the same race exists with inline updates.
"""

import logging
import queue
import threading

from ai_runtime.thread_stages import SHUTDOWN_SENTINEL

from rl_core.protocols.replay_buffer_protocol import ReplayBufferProtocol

logger = logging.getLogger(__name__)


class PriorityUpdater:
    """Updater thread: priority queue → ``buffer.update_priorities``."""

    # Queue poll interval; bounds shutdown latency while idle.
    _QUEUE_POLL_TIMEOUT: float = 0.1

    def __init__(
        self,
        replay_buffer: ReplayBufferProtocol,
        priority_queue: queue.Queue,
        shutdown_event: threading.Event,
    ) -> None:
        """Bind the buffer and the priority queue."""
        self._buffer = replay_buffer
        self._priority_queue = priority_queue
        self._shutdown_event = shutdown_event

    def run(self) -> None:
        """Updater thread main loop: apply updates until shutdown / sentinel."""
        logger.info("PriorityUpdater.run started")
        n_applied = 0
        while not self._shutdown_event.is_set():
            try:
                item = self._priority_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if item is SHUTDOWN_SENTINEL:
                break
            if self._apply(item):
                n_applied += 1

        # Drain remaining updates — cheap, and keeps priorities as fresh
        # as possible for a resumed run's buffer snapshot.
        while True:
            try:
                item = self._priority_queue.get_nowait()
            except queue.Empty:
                break
            if item is SHUTDOWN_SENTINEL:
                continue
            if self._apply(item):
                n_applied += 1

        logger.info("PriorityUpdater.run exiting: %d updates applied", n_applied)

    def _apply(self, item) -> bool:
        """Apply one ``(indices, td_errors)`` pair; log and drop on error."""
        indices, td_errors = item
        try:
            self._buffer.update_priorities(indices, td_errors)
        except Exception as exc:
            logger.exception(
                "PriorityUpdater: update_priorities failed (%s); dropped", exc,
            )
            return False
        return True
