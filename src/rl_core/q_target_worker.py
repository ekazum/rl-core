"""``QTargetWorker`` — sole owner of ``q_target``, serving target forwards and syncs.

No other code forwards ``q_target`` or loads weights into it. ``q_target`` can
live on its own device, so each update's target forward runs there while the
learner runs its ``q_online`` forwards: the learner submits its request first
and awaits the future only when it needs the values.

Two request types share the target queue:

* a forward request (``TargetRequest``, or any object with a ``response``
  future that ``decode`` understands) → ``decode`` builds the input on the
  target device → no-grad ``q_target`` forward → ``(B, A)`` values returned on
  ``learner_device``.
* ``SyncRequest`` — a ``q_online.state_dict()`` snapshot; ``load_state_dict``
  does the (cross-device) copy. No future: the learner does not wait on syncs.
"""

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import torch
from ai_runtime.pipeline import SHUTDOWN_SENTINEL
from torch import nn

from rl_core.sync_request import SyncRequest

logger = logging.getLogger(__name__)


class QTargetWorker:
    """Owns ``q_target``: serves forward requests and applies sync snapshots."""

    # How long ``run()`` blocks on the queue between shutdown_event checks.
    _QUEUE_POLL_TIMEOUT: float = 0.1

    def __init__(
        self,
        q_target: nn.Module,
        target_queue: queue.Queue,
        learner_device: torch.device,
        shutdown_event: threading.Event,
        decode: Callable[[Any], torch.Tensor] | None = None,
        name: str = "qtarget-worker",
    ) -> None:
        """Bind ``q_target`` (already on its device), the queue and the result device.

        ``decode(request)`` returns the ``(B, D)`` input on ``q_target``'s device;
        the default moves ``request.inputs`` there.
        """
        self._q_target = q_target
        self._target_queue = target_queue
        self._learner_device = learner_device
        self._shutdown_event = shutdown_event
        self._decode = decode if decode is not None else self._default_decode
        self._name = name
        self._device: torch.device = next(q_target.parameters()).device

    def _default_decode(self, request) -> torch.Tensor:
        """``request.inputs`` as a tensor on ``q_target``'s device."""
        return torch.as_tensor(request.inputs).to(self._device, non_blocking=True)

    def run(self) -> None:
        """Worker thread main loop: dispatch forward and sync requests.

        Exits on ``shutdown_event`` or a ``SHUTDOWN_SENTINEL``. On exit, fails the
        queued forward requests' futures so the learner never hangs on
        ``result()``, and drops queued syncs.
        """
        logger.info(
            "%s: run started — q_target on %s, results to %s",
            self._name, self._device, self._learner_device,
        )
        n_forwards = 0
        n_syncs = 0
        while not self._shutdown_event.is_set():
            try:
                item = self._target_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                continue
            if item is SHUTDOWN_SENTINEL:
                logger.info(
                    "%s: received SHUTDOWN_SENTINEL after %d forwards / %d syncs",
                    self._name, n_forwards, n_syncs,
                )
                break
            if isinstance(item, SyncRequest):
                self._q_target.load_state_dict(item.state_dict)
                n_syncs += 1
                continue
            try:
                inputs = self._decode(item)
                with torch.no_grad():
                    values = self._q_target(inputs)             # (B, A) on target device
                item.response.set_result(values.to(self._learner_device))
            except BaseException as exc:
                logger.exception(
                    "%s: target forward raised; propagating to learner future",
                    self._name,
                )
                item.response.set_exception(exc)
            n_forwards += 1

        n_failed = 0
        while True:
            try:
                leftover = self._target_queue.get_nowait()
            except queue.Empty:
                break
            if leftover is SHUTDOWN_SENTINEL or isinstance(leftover, SyncRequest):
                continue
            leftover.response.set_exception(
                RuntimeError("QTargetWorker shutting down; pending request rejected")
            )
            n_failed += 1

        logger.info(
            "%s: run exiting (%d forwards, %d syncs, rejected %d pending, "
            "shutdown_event=%s)",
            self._name, n_forwards, n_syncs, n_failed, self._shutdown_event.is_set(),
        )
