"""``PrioritizedReplayBuffer`` — thread-safe prioritized replay over a ``ColumnStore``.

The project's codec decides what a row holds; the buffer owns priorities, the
``SumTree`` index, the β anneal, the write cursor and the lock. New rows enter
at the running max priority (Schaul et al.), so each is drawn at least once
before a TD error refines it. ``alpha=0`` gives uniform sampling.

Concurrency is an ``ai_runtime`` ``ReadWriteLock``: ``draw_block``,
``gather_into``, ``__len__`` and ``get_priority_stats`` share it;
``push_batch`` and ``update_priorities`` take it exclusively. Clients:
``ReplayWriter`` → ``push_batch``, ``ReplayReader`` → ``draw_block``, the
learner → ``gather_into``, ``PriorityUpdater`` → ``update_priorities``,
``StatsProcessor`` → ``get_priority_stats``. Given a ``stats_q``, the buffer
emits ``DrawEvent`` / ``OverwriteEvent`` off the lock and drops them on a
full queue.
"""

import logging
import queue

import numpy as np
from ai_runtime.rw_lock import ReadWriteLock

from rl_core.column_store import ColumnStore
from rl_core.events.draw_event import DrawEvent
from rl_core.events.overwrite_event import OverwriteEvent
from rl_core.protocols.row_codec_protocol import RowCodecProtocol
from rl_core.sum_tree import SumTree

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    """Prioritized experience replay: columnar rows, sum-tree draws, RW lock."""

    def __init__(
        self,
        capacity: int,
        codec: RowCodecProtocol,
        alpha: float,
        beta_is_start: float,
        n_anneal: int,
        priority_floor: float,
        seed_seq: np.random.SeedSequence,
        stats_q: "queue.Queue | None" = None,
    ) -> None:
        """Allocate the codec's columns and bind the PER hyperparameters.

        ``priority = (|td| + priority_floor) ** alpha``; β rises linearly from
        ``beta_is_start`` to 1 over ``n_anneal`` drawn batches.
        """
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1; got {capacity}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        if not (0.0 <= beta_is_start <= 1.0):
            raise ValueError(f"beta_is_start must be in [0, 1]; got {beta_is_start}")
        if n_anneal < 1:
            raise ValueError(f"n_anneal must be >= 1; got {n_anneal}")
        if priority_floor <= 0:
            raise ValueError(f"priority_floor must be > 0; got {priority_floor}")

        self._capacity = capacity
        self._codec = codec
        self._alpha = alpha
        self._beta_is_start = beta_is_start
        self._beta_is = beta_is_start
        self._n_anneal = n_anneal
        self._priority_floor = priority_floor
        # Single draw consumer (ReplayReader), so a private generator suffices.
        self._rng = np.random.default_rng(seed_seq)

        self._store = ColumnStore(capacity, codec.columns())
        # Rows at index >= _size hold priority 0 and are unreachable by sampling.
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._tree = SumTree(capacity)

        self._cursor = 0
        self._size = 0
        # Running max of written priorities; 0 until the first update, so the
        # first pushes enter at 1.0.
        self._max_priority_seen = 0.0
        self._update_count = 0
        self._eviction_count = 0
        self._stats_q = stats_q
        self._lock = ReadWriteLock()

    @property
    def capacity(self) -> int:
        """Rows the buffer holds when full."""
        return self._capacity

    @property
    def codec(self) -> RowCodecProtocol:
        """The codec the columns came from."""
        return self._codec

    def _emit(self, event) -> None:
        """Best-effort enqueue of a stats event; drop on a full queue or no queue."""
        if self._stats_q is None:
            return
        try:
            self._stats_q.put_nowait(event)
        except queue.Full:
            pass

    def push_batch(self, batch) -> None:
        """Encode ``batch`` (off the lock) and store its rows at max priority."""
        length = batch.num_rows
        if length < 1:
            raise ValueError("push_batch requires at least one row")
        if length > self._capacity:
            raise ValueError(f"batch has {length} rows; capacity is {self._capacity}")
        rows = self._codec.encode(batch)
        missing = set(self._store.names) - set(rows)
        if missing:
            raise ValueError(f"codec.encode left out columns {sorted(missing)}")

        with self._lock.write_lock():
            logger.debug("buffer: push_batch holds write-lock (ingest)")
            priority = self._max_priority_seen if self._max_priority_seen > 0 else 1.0
            first = min(length, self._capacity - self._cursor)
            lo, hi = self._cursor, self._cursor + first
            self._eviction_count += max(0, min(hi, self._size) - lo)
            self._store.write(slice(lo, hi), rows, slice(0, first))
            self._priorities[lo:hi] = priority
            ranges = [(lo, hi)]
            rest = length - first
            if rest > 0:  # wraparound — second slice at the front
                self._eviction_count += min(rest, self._size)
                self._store.write(slice(0, rest), rows, slice(first, length))
                self._priorities[0:rest] = priority
                ranges.append((0, rest))
            slots = np.arange(lo, hi)
            if rest > 0:
                slots = np.concatenate([slots, np.arange(rest)])
            self._tree.update(slots, np.full(slots.shape[0], priority))
            self._size = min(self._size + length, self._capacity)
            self._cursor = (self._cursor + length) % self._capacity
        self._emit(OverwriteEvent(tuple(ranges)))

    def draw_block(
        self, batch_size: int, n_batches: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Draw ``n_batches`` prioritized ``(indices, is_weights)`` pairs per build.

        Stratified: one uniform per equal-probability segment of each batch
        (Schaul et al. App. B.2.1), mapped to slots by a sum-tree descent. β
        advances once per batch, the same schedule as K single draws.
        """
        if n_batches < 1:
            raise ValueError(f"n_batches must be >= 1; got {n_batches}")

        with self._lock.read_lock():
            logger.debug("buffer: draw_block holds read-lock (sample)")
            n = self._size
            if n < batch_size:
                raise ValueError(
                    f"buffer has {n} transitions; cannot sample batch_size={batch_size}"
                )
            total = self._tree.total
            if total <= 0:
                raise RuntimeError("all priorities are zero; cannot sample")

            segment = total / batch_size
            offsets = np.arange(batch_size) * segment                  # (B,)
            uniforms = (
                self._rng.uniform(0.0, segment, size=(n_batches, batch_size))
                + offsets[None, :]
            ).ravel()
            all_indices = self._tree.sample(uniforms)
            # Float rounding can land a uniform exactly on total → index n.
            np.minimum(all_indices, n - 1, out=all_indices)

            draws: list[tuple[np.ndarray, np.ndarray]] = []
            sample_probs_all = np.asarray(self._priorities[all_indices] / total)
            for k in range(n_batches):
                lo = k * batch_size
                indices = all_indices[lo:lo + batch_size]
                weights = (n * sample_probs_all[lo:lo + batch_size]) ** (-self._beta_is)
                weights = (weights / weights.max()).astype(np.float32)
                draws.append((indices, weights))
                self._update_count += 1
                frac = min(1.0, self._update_count / self._n_anneal)
                self._beta_is = self._beta_is_start + (1.0 - self._beta_is_start) * frac
        self._emit(DrawEvent(all_indices))
        return draws

    def gather_into(self, indices: np.ndarray, out: dict[str, np.ndarray]) -> None:
        """Take rows ``indices`` of the columns named in ``out`` into its arrays.

        ``out`` may name any subset of the columns; its arrays can be views of
        pinned memory. Shared lock, so a push cannot tear the read.
        """
        with self._lock.read_lock():
            self._store.take(indices, out)

    def gather(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        """Allocating ``gather_into`` over every column."""
        out = {
            name: np.empty((indices.shape[0], *self._store.column(name).shape[1:]),
                           dtype=self._store.column(name).dtype)
            for name in self._store.names
        }
        self.gather_into(indices, out)
        return out

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Set ``priority = (|td| + priority_floor) ** alpha`` for the given rows.

        Duplicate indices resolve last-write-wins.
        """
        idx = np.asarray(indices, dtype=np.int64)
        td = np.asarray(td_errors, dtype=np.float64)
        if idx.shape != td.shape:
            raise ValueError(
                f"indices {idx.shape} and td_errors {td.shape} must have the same shape"
            )
        priorities = (np.abs(td) + self._priority_floor) ** self._alpha
        with self._lock.write_lock():
            self._priorities[idx] = priorities
            self._tree.update(idx, priorities)
            peak = float(priorities.max())
            if peak > self._max_priority_seen:
                self._max_priority_seen = peak

    def __len__(self) -> int:
        """Rows currently stored."""
        with self._lock.read_lock():
            return self._size

    def get_priority_stats(self) -> dict:
        """Priority spread (``prio_*``, normalized entropy), β and occupancy."""
        with self._lock.read_lock():
            n = self._size
            if n == 0:
                return {
                    "prio_min": 0.0, "prio_mean": 0.0, "prio_median": 0.0,
                    "prio_max": 0.0, "prio_p99": 0.0, "prio_max_over_mean": 0.0,
                    "prio_entropy": 0.0,
                    "beta_is": self._beta_is, "size": 0, "fill_frac": 0.0,
                    "eviction_count": self._eviction_count,
                }
            prio = self._priorities[:n]
            total = float(prio.sum())
            probs = prio / total
            entropy = float(-(probs * np.log(probs)).sum() / np.log(n)) if n > 1 else 0.0
            mean_p = float(prio.mean())
            return {
                "prio_min": float(prio.min()),
                "prio_mean": mean_p,
                "prio_median": float(np.median(prio)),
                "prio_max": float(prio.max()),
                "prio_p99": float(np.percentile(prio, 99)),
                "prio_max_over_mean": float(prio.max() / mean_p) if mean_p > 0 else 0.0,
                "prio_entropy": entropy,
                "beta_is": self._beta_is,
                "size": n,
                "fill_frac": n / self._capacity,
                "eviction_count": self._eviction_count,
            }
