"""``QNetworkWorker`` — no-grad ``q_online`` inference server for the actor pipeline.

Actor threads put ``QNetRequest``s on a shared bounded queue and block on
the response future; a worker
pulls, forwards on ``q_online`` under ``torch.no_grad()``, and fulfils
the future with the ``(L, A)`` q-values. With several processors
waiting, a worker concatenates pending requests into one forward while
their combined rows fit its staging buffer.

Half-lock pattern
-----------------
Workers read ``q_online`` without acquiring ``network_lock``. Occasional
inconsistent reads during the learner's ``optimizer.step`` produce
slightly noisy Q-values that ε-greedy exploration dominates. Acquiring
the lock would serialise inference against training and lose the
concurrency the thread groups exist for.

Staging buffer
--------------
Each worker preallocates one ``(max_rows, D)`` staging tensor — pinned
when the device is CUDA — so the host-side copy is a single memcpy and
the H2D transfer is ``non_blocking``. ``max_rows`` is the actors' batch
size: no single request can exceed it.
"""

import logging
import queue
import threading

import torch
from ai_runtime.pipeline import SHUTDOWN_SENTINEL
from torch import nn

from rl_core.q_net_request import QNetRequest

logger = logging.getLogger(__name__)


class QNetworkWorker:
    """Drains ``QNetRequest``s from a shared queue; one no-grad forward per drain."""

    # How long ``run()`` blocks on the queue between shutdown_event checks.
    _QUEUE_POLL_TIMEOUT: float = 0.1

    def __init__(
        self,
        q_online: nn.Module,
        request_queue: queue.Queue,
        shutdown_event: threading.Event,
        state_dim: int,
        max_rows: int,
        name: str = "qnet-worker",
    ) -> None:
        """Bind the shared ``q_online`` and preallocate the staging buffer.

        ``q_online`` is the live training network on the learner's GPU —
        read lock-free (half-lock pattern, module docstring). ``max_rows``
        bounds a single request and the coalescing window; sized to the
        actors' batch size.
        """
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1; got {state_dim}")
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1; got {max_rows}")
        self._q_online = q_online
        self._request_queue = request_queue
        self._shutdown_event = shutdown_event
        self._max_rows = max_rows
        self._name = name
        self._device: torch.device = next(q_online.parameters()).device
        # Pinned host staging enables async H2D; plain host tensor on CPU runs.
        self._staging = torch.empty(
            (max_rows, state_dim),
            dtype=torch.float32,
            pin_memory=(self._device.type == "cuda"),
        )
        # A request pulled during coalescing that didn't fit the staging
        # window; served first on the next loop iteration.
        self._held: QNetRequest | None = None

    def run(self) -> None:
        """Worker thread main loop: drain requests, forward, fulfil futures.

        Exits when ``shutdown_event`` is set (checked between pulls) or a
        ``SHUTDOWN_SENTINEL`` is pulled. On exit, drains remaining
        requests and fails their futures so no processor hangs on
        ``result()``.
        """
        logger.info(
            "%s: run started on device %s (staging %dx%d, pinned=%s)",
            self._name, self._device, *self._staging.shape,
            self._staging.is_pinned() if self._device.type == "cuda" else False,
        )
        n_served = 0
        got_sentinel = False
        while not self._shutdown_event.is_set() and not got_sentinel:
            requests, got_sentinel = self._collect()
            if not requests:
                continue
            self._serve(requests)
            n_served += len(requests)

        n_failed = self._drain_and_fail()
        logger.info(
            "%s: run exiting (served %d requests, rejected %d pending, "
            "shutdown_event=%s)",
            self._name, n_served, n_failed, self._shutdown_event.is_set(),
        )

    def _collect(self) -> tuple[list[QNetRequest], bool]:
        """Gather one or more requests whose combined rows fit the staging buffer.

        Blocks up to ``_QUEUE_POLL_TIMEOUT`` for the first request (the
        held-over one, if any, goes first), then drains further pending
        requests non-blockingly while they fit. Returns
        ``(requests, got_sentinel)``.
        """
        if self._held is not None:
            first, self._held = self._held, None
        else:
            try:
                item = self._request_queue.get(timeout=self._QUEUE_POLL_TIMEOUT)
            except queue.Empty:
                return [], False
            if item is SHUTDOWN_SENTINEL:
                return [], True
            first = item

        requests = [first]
        rows = first.vectors.shape[0]
        while rows < self._max_rows:
            try:
                item = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if item is SHUTDOWN_SENTINEL:
                return requests, True
            if rows + item.vectors.shape[0] > self._max_rows:
                self._held = item  # doesn't fit — serve it next round
                break
            requests.append(item)
            rows += item.vectors.shape[0]
        return requests, False

    def _serve(self, requests: list[QNetRequest]) -> None:
        """One staging copy + H2D + no-grad forward; split results per request."""
        try:
            offset = 0
            slices: list[tuple[int, int]] = []
            for req in requests:
                length = req.vectors.shape[0]
                if length > self._max_rows:
                    raise ValueError(
                        f"request has {length} rows; staging holds {self._max_rows}"
                    )
                self._staging[offset:offset + length].copy_(
                    torch.from_numpy(req.vectors)
                )
                slices.append((offset, offset + length))
                offset += length

            x = self._staging[:offset].to(self._device, non_blocking=True)
            with torch.no_grad():
                q_vals = self._q_online(x)                      # (ΣL, A)
            out = q_vals.cpu().numpy()                          # syncs the transfer

            if len(requests) == 1:
                requests[0].response.set_result(out)
            else:
                for req, (lo, hi) in zip(requests, slices):
                    req.response.set_result(out[lo:hi].copy())
        except BaseException as exc:  # noqa: BLE001
            logger.exception(
                "%s: forward raised; propagating to %d caller future(s)",
                self._name, len(requests),
            )
            for req in requests:
                if not req.response.done():
                    req.response.set_exception(exc)

    def _drain_and_fail(self) -> int:
        """Fail the held request and everything still queued; return the count."""
        n_failed = 0
        leftovers: list[QNetRequest] = []
        if self._held is not None:
            leftovers.append(self._held)
            self._held = None
        while True:
            try:
                item = self._request_queue.get_nowait()
            except queue.Empty:
                break
            if item is SHUTDOWN_SENTINEL:
                continue
            leftovers.append(item)
        for req in leftovers:
            req.response.set_exception(
                RuntimeError("QNetworkWorker shutting down; pending request rejected")
            )
            n_failed += 1
        return n_failed
