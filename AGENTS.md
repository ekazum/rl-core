# rl-core — notes for AI coding tools

rl-core gives the threads and data types of a reinforcement learning training pipeline. A project builds an agent by
writing its environment loop and the few objects named in the protocols below, then wiring rl-core's parts with
queues and one shared `threading.Event`.

Install: `pip install "rl-core @ git+https://github.com/ekazum/rl-core.git@v0.1.1"` (pulls `ai-runtime`, numpy,
torch). Everything below is importable from the top level: `from rl_core import ...`.

## What rl-core provides, what the project writes

| Area | rl-core provides | The project writes |
|---|---|---|
| PPO | `PPOPolicyNetwork`, `PPORolloutBatch` (GAE), `PPOUpdateWorker` (thread), `PPOCheckpointManager` | the environment loop that fills `PPORolloutBatch` |
| Prioritized replay | `SumTree`, `ReplayWriter`, `ReplayReader`, `PriorityUpdater`, `StatsProcessor` | a buffer meeting `ReplayBufferProtocol`, a telemetry sink meeting `ReplayTelemetryProtocol` |
| Value-network inference | `QNetworkWorker`, `QNetRequest` | the actors that send requests, the learner that trains the network |
| Checkpoints | `CheckpointWriter`, `CheckpointRequest`, `LatestRequest`, `CheckpointState` | a store meeting `SnapshotStoreProtocol` |
| Stats events | `DrawEvent`, `OverwriteEvent`, `CheckpointEvent` | the buffer emits the first two |

## Protocols (`rl_core.protocols`)

```python
class ReplayBufferProtocol(Protocol):
    def push_batch(self, batch) -> None: ...            # batch.num_rows transitions
    def draw_block(self, batch_size: int, n_batches: int) -> list[tuple[np.ndarray, np.ndarray]]: ...
                                                        # n_batches (indices, is_weights) draws
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None: ...
    def get_priority_stats(self) -> dict: ...           # must contain "size"
    def __len__(self) -> int: ...

class ReplayTelemetryProtocol(Protocol):
    @property
    def transitions_pushed(self) -> int: ...
    def note_transitions(self, n: int) -> None: ...
    def log_buffer_stats(self, episode_index: int, metrics: dict) -> None: ...

class SnapshotStoreProtocol(Protocol):
    def write_snapshot(self, update: int, q_online_state_dict: dict, dims: dict) -> Path: ...
    def write_latest(self, state) -> None: ...          # a CheckpointState
    def delete_snapshot(self, path: Path) -> None: ...
```

`SumTree(capacity)` (`build`, `update`, `sample`, `total`) is the usual base for a `draw_block` with priorities.

## Threads and their queues

All take `shutdown_event: threading.Event`. `PPOUpdateWorker` is a `threading.Thread`; the others are plain classes
with `run()`; start them with `ai_runtime.thread_handle.ThreadHandle(target=obj.run, name=...)`, which re-raises a
thread's exception on `join`.

| Thread | Reads | Writes |
|---|---|---|
| `ReplayWriter(replay_buffer, write_queue, telemetry, shutdown_event)` | transition batches (`num_rows`) | `buffer.push_batch`, `telemetry.note_transitions` |
| `ReplayReader(replay_buffer, prefetch_queue, batch_size, min_buffer_size, block_batches, shutdown_event)` | `buffer.draw_block` | `(indices, is_weights)` items on `prefetch_queue` |
| `PriorityUpdater(replay_buffer, priority_queue, shutdown_event)` | `(indices, td_errors)` items | `buffer.update_priorities` |
| `StatsProcessor(stats_q, replay_buffer, telemetry, capacity, shutdown_event)` | `DrawEvent`, `OverwriteEvent`, `CheckpointEvent` | `telemetry.log_buffer_stats` |
| `QNetworkWorker(q_online, request_queue, shutdown_event, state_dim, max_rows)` | `QNetRequest(vectors, response=Future())` | `(L, A)` outputs into each future |
| `CheckpointWriter(write_queue, stats_q, checkpoint_manager, shutdown_event, keep_all)` | `CheckpointRequest`, `LatestRequest` | the store; `CheckpointEvent` on `stats_q` |
| `PPOUpdateWorker(model, optimizer, rollout_queue, device, shutdown_event, epochs=4, minibatch_size=256, ...)` | `PPORolloutBatch` via `submit` | the model's weights |

## Assembly order

PPO (see `examples/cartpole_ppo/train.py`):

1. `model = PPOPolicyNetwork(obs_dim, n_actions)`; an optimizer over its parameters.
2. `updater = PPOUpdateWorker(model, optimizer, queue.Queue(maxsize=1), device, shutdown)`; `updater.start()`.
3. Each iteration: collect steps into `PPORolloutBatch` pieces → `PPORolloutBatch.concatenate(parts)` →
   `batch.compute_gae(gamma, gae_lambda)` → `updater.submit(batch)` → `updater.wait_idle()`.
4. Optional: `PPOCheckpointManager(dir).save_snapshot(model, updater.updates, meta)`.
5. `shutdown.set()`; `updater.join()`.

Prioritized replay with a value network:

1. Create the buffer, telemetry, store, and queues: `write_queue`, bounded `prefetch_queue`, `priority_queue`,
   `stats_q`, `request_queue`, checkpoint `write_queue`.
2. Start `StatsProcessor` and `CheckpointWriter`, then `ReplayWriter`, `PriorityUpdater`, `ReplayReader`, then the
   `QNetworkWorker`s, then the actors and the learner.
3. The learner takes `(indices, is_weights)` from `prefetch_queue`, gathers the rows from the buffer, trains, puts
   `(indices, |td|)` on `priority_queue`, and puts `CheckpointRequest(update, cpu_state_dict, dims)` on the checkpoint
   queue.
4. Stop: `shutdown.set()`, put `ai_runtime.pipeline.SHUTDOWN_SENTINEL` on each queue, join producers before consumers
   (actors, reader, learner, then writer, updater, checkpoint writer, stats). `ReplayWriter` and `CheckpointWriter`
   drain their queues before exiting.

## Rules that are easy to break

- PPO is on-policy: call `updater.wait_idle()` after `submit` and before collecting the next rollout.
- Collect with the model in eval mode and under `torch.no_grad()`; the stored log-probs must match what the update
  recomputes.
- Use `PPOPolicyNetwork.masked_log_probs(logits, invalid)` on both sides; `invalid` is True for a masked action.
  Numpy code masking the same way uses `MASKED_LOGIT`.
- `PPORolloutBatch.rows` is an episode id. Give an env a new id when its episode ends and when the rollout ends, so
  GAE never joins two segments.
- `forward` returns `(B, A + 1)`; split with `PPOPolicyNetwork.split(out)` into logits and value.
- The buffer emits `DrawEvent` / `OverwriteEvent` with `put_nowait` and drops them on a full `stats_q`; the
  `CheckpointEvent` from `CheckpointWriter` is never dropped.
- `CheckpointRequest.state_dict` is a CPU copy taken by the learner; the writer thread only writes it.
- `ReplayReader` waits until `len(buffer) >= max(min_buffer_size, batch_size)` before the first draw.

## Examples

- `examples/cartpole_ppo/train.py` — PPO on gymnasium CartPole, CPU, seeded. `pip install "rl-core[examples]"`,
  then `python examples/cartpole_ppo/train.py`.

## The four repos

| Repo | Gives | Depends on |
|---|---|---|
| [ai-runtime](https://github.com/ekazum/ai-runtime) | threads, `SHUTDOWN_SENTINEL`, `Pipeline`, GPU batching, service supervision, SQLite stores, checkpoint managers, `YamlConfig` | standard library; `[torch]`, `[config]` |
| [ai-llm](https://github.com/ekazum/ai-llm) | an LLM generation service: engine, GPU worker, server, client, stats store | ai-runtime |
| [ai-ui](https://github.com/ekazum/ai-ui) | a Flask web shell and a runs dashboard over a SQLite runs DB | ai-runtime |
| rl-core | the RL threads and data types above | ai-runtime |
