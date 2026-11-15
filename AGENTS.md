# rl-core — notes for AI coding tools

rl-core gives the threads and data types of a reinforcement learning training pipeline. A project builds an agent by
writing its environment loop and the few objects named in the protocols below, then wiring rl-core's parts with
queues and one shared `threading.Event`.

Install: `pip install "rl-core @ git+https://github.com/ekazum/rl-core.git@v0.10.0"` (pulls `ai-runtime` with its
config extra, numpy, torch). Everything below is importable from the top level: `from rl_core import ...`.

## What rl-core provides, what the project writes

| Area | rl-core provides | The project writes |
|---|---|---|
| PPO | `PPOPolicyNetwork`, `PPORolloutBatch` (GAE), `PPOUpdateWorker` (thread), `PPOTrainer` (loop), `PPOCheckpointManager` | the environment loop that fills `PPORolloutBatch` (the `collect` hook), optional `report`, `after_update`, `finish` hooks |
| Prioritized replay | `PrioritizedReplayBuffer`, `ColumnStore`, `DenseCodec`, `SumTree`, `ReplayWriter`, `ReplayReader`, `PriorityUpdater`, `StatsProcessor` | the batch type (with `num_rows`), a codec meeting `RowCodecProtocol` or a `DenseCodec` spec, a telemetry sink meeting `ReplayTelemetryProtocol`, the learner |
| Value-network inference | `QNetworkWorker`, `QNetRequest`, `QTargetWorker`, `TargetRequest`, `SyncRequest` | the network (any `nn.Module` mapping `(B, D)` to `(B, A)`), the actors that send requests |
| DQN training | `DQNLearner` (Double DQN loop, LR schedule, sync and checkpoint cadences), `DQNBatch` | the optimizer, the `prepare` and `make_target_request` hooks, a telemetry sink meeting `LearnerTelemetryProtocol` |
| DDPG training | `DDPGLearner` (abstract critic loop: step, soft target update, LR schedules, snapshot cadence, `max_updates` stop; optional actor step through the frozen critic), `CriticStep`, `GradientExchange` (gradient averaging between one learner per critic copy) | the subclass's `_critic_backward` (forward and backward, returns a `CriticStep`), `_actor_backward` with an actor, optional `_release`, `_on_step`, `_snapshot_state`; the two critics and the optimizers |
| Checkpoints | `CheckpointWriter`, `CheckpointRequest`, `LatestRequest`, `CheckpointState` | a store meeting `SnapshotStoreProtocol` |
| Stats events | `DrawEvent`, `OverwriteEvent`, `CheckpointEvent` | the buffer emits the first two |
| Evaluation | `EvalTask`, `ScoreTask` (abstract), `EvalConfig`, `EvalStore` (`eval.db`) | the task subclasses that produce the policy's outputs and rewards, a config subclass with its own fields, the manager that calls the tasks |
| Actor feeding | `RowSampler`, `ActorFeeder` (thread) | a `RowSampler` subclass that knows the table's data, the actor threads that take the id batches, a telemetry sink meeting `FeederTelemetryProtocol` |

## Protocols (`rl_core.protocols`)

```python
class RowCodecProtocol(Protocol):
    def columns(self) -> dict[str, tuple[tuple[int, ...], np.dtype | type]]: ...
                                                        # name -> (row_shape, dtype)
    def encode(self, batch) -> dict[str, np.ndarray]: ...  # one (num_rows, *row_shape) array per column

class ReplayBufferProtocol(Protocol):                   # PrioritizedReplayBuffer meets it
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

class LearnerTelemetryProtocol(Protocol):
    @property
    def transitions_pushed(self) -> int: ...             # read for the push-ratio throttle
    def note_update(self, count: int) -> None: ...      # the update count, after every step

class FeederTelemetryProtocol(Protocol):
    @property
    def transitions_pushed(self) -> int: ...
    @property
    def update_count(self) -> int: ...                  # both read for the push-per-update pace

class SnapshotStoreProtocol(Protocol):
    def write_snapshot(self, update: int, q_online_state_dict: dict, dims: dict) -> Path: ...
    def write_latest(self, state) -> None: ...          # a CheckpointState
    def delete_snapshot(self, path: Path) -> None: ...
```

## The replay buffer

```python
PrioritizedReplayBuffer(capacity, codec, alpha, beta_is_start, n_anneal, priority_floor,
                        seed_seq: np.random.SeedSequence, stats_q: queue.Queue | None = None)
buffer.push_batch(batch)                     # codec.encode(batch) off the lock, rows at max priority
buffer.draw_block(batch_size, n_batches)     # [(indices, is_weights), ...]
buffer.gather(indices) -> dict[str, np.ndarray]           # every column, allocated
buffer.gather_into(indices, out: dict[str, np.ndarray])   # named columns into your arrays
buffer.update_priorities(indices, td_errors) # (|td| + priority_floor) ** alpha
DenseCodec({"obs": ((4,), np.float32), "action": ((), np.int64), ...})  # stores batch.<name> as-is
```

`alpha=0` gives uniform replay. Rows are fixed-shape numpy columns in CPU memory and are drawn one at a time;
n-step returns and sequence sampling are the project's. A project with a compact row layout writes its own codec;
`ColumnStore` and `SumTree(capacity)` are also usable alone.

## The DQN learner

```python
DQNLearner(q_online, optimizer, batch_queue, target_queue, priority_queue, write_queue | None, telemetry,
           network_lock, shutdown_event, *,
           prepare,               # (queued item) -> DQNBatch on the learner device
           make_target_request,   # (queued item, Future) -> the QTargetWorker request for its next states
           gamma, lr_init, lr_min, lr_decay=1.0, lr_schedule="exponential" | "linear",
           max_updates=0,         # 0 = no cap
           target_sync_every, checkpoint_every=0, min_checkpoint_update=0, latest_every=0,
           push_per_update_ratio=0.0, layout_dims=None, state_builder=None,
           release=None,          # (queued item) -> None, after every batch
           clip_next_q=None,      # (B,) target values -> bounded values
           on_step=None)          # (dict: update, loss, td_error, q_values, q_taken, target, next_q_value, dones, lr)
DQNBatch(states, next_states, actions, rewards, dones, is_weights, indices, durations=None)
learner.get_state() / learner.load_state(state)   # optimizer + update counters, for resume
```

`durations` gives SMDP targets `γ^(τ-1) · (r + γ · next_q)`; without it the target is one-step. The settings are
the project's values; rl-core has no config file.

## The DDPG critic learner

```python
class MyLearner(DDPGLearner):
    def _critic_backward(self, batch) -> CriticStep:   # forward + loss.backward(); no optimizer step
        ...
    # optional: _release(batch), _on_step(update, metrics), _snapshot_state() -> dict

MyLearner(critic, critic_target, optimizer, batch_queue, priority_queue, write_queue | None, telemetry,
          shutdown_event, *, tau, lr_init, lr_min, max_updates,
          checkpoint_every=0, push_per_update_ratio=0.0, snapshot_dims=None, exchange=None, rank=0,
          actor_optimizer=None, actor_lr_init=0.0, actor_lr_min=0.0, actor_every=1, actor_start_update=0)
    # with an actor_optimizer: _actor_backward(batch) -> dict, backward of −Q(s, μ(s)) into the actor
CriticStep(indices, td_abs, metrics={})
learner.done      # threading.Event, set once the final snapshot is queued
learner.primary   # rank == 0
GradientExchange(critics, shutdown_event, check_every=0)   # critics[r] is rank r's copy
exchange.average(rank) -> bool   # False once shutdown_event is set while waiting
```

Per update: the learning rate is set (linear from `lr_init` to `lr_min` over `max_updates`), `_critic_backward`
runs, the optimizer steps (no clipping; `grad_norm` is measured), and `θ' ← (1 − τ) · θ' + τ · θ`. Then
`(indices, td_abs)` goes on `priority_queue`. With an `actor_optimizer`, from update `actor_start_update` every
`actor_every` updates, the actor step runs on the same batch: the critic's weights are frozen, `_actor_backward`
back-propagates −Q(s, μ(s)) into the actor, and the actor optimizer steps (its LR linear from `actor_lr_init` to
`actor_lr_min`). `_on_step` gets the metrics plus `grad_norm` and `lr`, and on actor updates the actor's metrics plus
`actor_grad_norm` and `actor_lr`. With
`push_per_update_ratio > 0`, an update waits until `(updates + 1) · ratio` transitions are stored. A failure sets
`shutdown_event` and re-raises; reaching `max_updates` does not set it.

Several critic copies (one per device): build one learner per copy, each with its own critic, target critic,
optimizer and batch queue, all holding one `GradientExchange` and their own `rank`. After the backward pass each
calls `average(rank)`: rank 0 sums the copies' gradients into its own, divides by the copy count and copies the mean
into the others, so every copy takes the same step. Give each learner an equal share of every drawn batch. Only rank
0 calls `note_update`; give the others no `write_queue`. Every `check_every` exchanges rank 0 compares the copies'
weight sums and copies its weights into a copy that differs.

## The PPO trainer

```python
PPOTrainer(worker: PPOUpdateWorker, shutdown_event, *,
           collect,               # () -> PPORolloutBatch from the current policy
           timesteps,             # transitions to collect
           max_updates=0,         # 0 = no cap
           gamma=1.0, gae_lambda=0.95,
           report=None,           # (index, rollout, steps) -> None, before each update
           after_update=None,     # (updates) -> bool, after each update; True stops
           finish=None,           # (updates, steps) -> None, after the loop, before the join
           join_timeout=60.0)
trainer.run() -> (updates, steps)  # starts the worker; on exit sets shutdown_event and joins it
PPOPolicyNetwork.masked_log_probs_np(logits, invalid)   # numpy twin of masked_log_probs, float32
PPOPolicyNetwork.sample(log_probs, rng, greedy)         # (B,) Gumbel-max draw, or argmax when greedy
```

`finish` runs after every stop, including a shutdown; check `shutdown_event` in it to skip work after a signal.

## Evaluation

```python
class EvalTask(ABC):                     # base of every evaluation task
    def run(self, *args, **kwargs) -> dict: ...      # abstract; the metrics row
class ScoreTask(EvalTask):
    def run(self, split: str) -> dict: ...           # abstract; the current policy on one split
EvalConfig.from_yaml(path)               # YamlConfig: batch_size, limit (0 = every row); subclass to add fields
EvalStore(eval_dir, criterion)           # eval.db; criterion names the dev column where higher is better
    .add_dev(epoch, checkpoint, metrics) # one dev_metrics row; epoch is the learner update count
    .add_test(metrics)  .test_metrics()  # key/value pairs
    .dev_rows()  .champion()             # rows by update; (update, checkpoint) of the best row, or None
    .selection_state()                   # (best criterion, rows since it), from ai-runtime's DevSelectionStore
```

The tasks define what a manager calls; a project subclasses them and its manager calls each. No task holds another.
`EvalStore` adds a project's metric columns to `dev_metrics` from the first row it is given.

## Actor feeding

```python
RowSampler(n_rows, seed_seq)             # rows 0 .. n_rows-1, a new permutation per pass from seed_seq
    .next() -> int                       # the next row; starts the next pass when this one is used up
    .pass_counter  .n_rows
    .get_snapshot()  .load_snapshot(d)   # pass counter and cursor; the pass order is regenerated from the seed
```

Subclass `RowSampler` to hold the table's data next to its rows; the feeder only calls `next()`.

## Threads and their queues

All take `shutdown_event: threading.Event`. `PPOUpdateWorker` is a `threading.Thread`; the others are plain classes
with `run()`; start them with `ai_runtime.thread_handle.ThreadHandle(target=obj.run, name=...)`, which re-raises a
thread's exception on `join`.

| Thread | Reads | Writes |
|---|---|---|
| `ReplayWriter(replay_buffer, write_queue, telemetry, shutdown_event)` | transition batches (`num_rows`) | `buffer.push_batch`, `telemetry.note_transitions` |
| `ReplayReader(replay_buffer, prefetch_queue, batch_size, min_buffer_size, block_batches, shutdown_event)` | `buffer.draw_block` | `(indices, is_weights)` items on `prefetch_queue` |
| `ActorFeeder(sampler, batch_queue, telemetry, batch_size, max_rows, push_per_update_ratio, shutdown_event)` | `sampler.next()`; the two counters of `FeederTelemetryProtocol` | `list[int]` of `batch_size` rows on `batch_queue`; stops after `max_rows` (0 = until shutdown); while `push_per_update_ratio > 0` and updates have started, waits while `transitions_pushed > ratio * update_count`; sets `shutdown_event` on a failure |
| `PriorityUpdater(replay_buffer, priority_queue, shutdown_event)` | `(indices, td_errors)` items | `buffer.update_priorities` |
| `StatsProcessor(stats_q, replay_buffer, telemetry, capacity, shutdown_event)` | `DrawEvent`, `OverwriteEvent`, `CheckpointEvent` | `telemetry.log_buffer_stats` |
| `QNetworkWorker(q_online, request_queue, shutdown_event, state_dim, max_rows)` | `QNetRequest(vectors, response=Future())` | `(L, A)` outputs into each future |
| `QTargetWorker(q_target, target_queue, learner_device, shutdown_event, decode=None, name=...)` | `TargetRequest(inputs, response=Future())`, `SyncRequest(state_dict)` | `(B, A)` target values on `learner_device` into each future; syncs load into `q_target`; queued futures fail at shutdown |
| `DQNLearner(q_online, optimizer, batch_queue, target_queue, priority_queue, write_queue, telemetry, network_lock, shutdown_event, *, ...)` | queued batches via `prepare`; target values | `(indices, \|td\|)` on `priority_queue`; `TargetRequest`s and `SyncRequest`s on `target_queue`; `CheckpointRequest`, `LatestRequest` on `write_queue`; sets `shutdown_event` on exit |
| `DDPGLearner` subclass `(critic, critic_target, optimizer, batch_queue, priority_queue, write_queue, telemetry, shutdown_event, *, ...)` | queued batches via `_critic_backward`; `transitions_pushed` for the throttle | `(indices, td_abs)` on `priority_queue`; `CheckpointRequest` on `write_queue`; `note_update` (rank 0 only); sets `done` on exit, `shutdown_event` on a failure |
| `CheckpointWriter(write_queue, stats_q, checkpoint_manager, shutdown_event, keep_all)` | `CheckpointRequest`, `LatestRequest` | the store; `CheckpointEvent` on `stats_q` |
| `PPOUpdateWorker(model, optimizer, rollout_queue, device, shutdown_event, epochs=4, minibatch_size=256, ...)` | `PPORolloutBatch` via `submit` | the model's weights |

## Assembly order

PPO (see `examples/cartpole_ppo/train.py`):

1. `model = PPOPolicyNetwork(obs_dim, n_actions)`; an optimizer over its parameters.
2. `updater = PPOUpdateWorker(model, optimizer, queue.Queue(maxsize=1), device, shutdown)`, not started.
3. `collect()`: step the envs into `PPORolloutBatch` pieces → `PPORolloutBatch.concatenate(parts)`.
4. Optional `after_update`: `PPOCheckpointManager(dir).save_snapshot(model, updater.updates, meta)`, scoring.
5. `PPOTrainer(updater, shutdown, collect=..., timesteps=..., gamma=..., gae_lambda=..., after_update=...).run()`;
   it computes the advantages, submits, waits for each update, then sets `shutdown` and joins the worker.

DQN with prioritized replay (see `examples/cartpole_dqn/train.py`):

1. A batch type with `num_rows`; `codec = DenseCodec({...})` naming its fields;
   `buffer = PrioritizedReplayBuffer(capacity, codec, ...)`; a telemetry sink.
2. Networks: `q_online` and `q_target` (same class, `q_target` loaded from `q_online`); a lock; an optimizer.
3. Queues: `write_queue`, bounded `prefetch_queue`, `priority_queue`, `target_queue`.
4. Hooks over the buffer: `make_target_request((indices, w), future)` gathers the next states into a
   `TargetRequest`; `prepare((indices, w))` gathers the rows into a `DQNBatch`.
5. `learner = DQNLearner(q_online, optimizer, prefetch_queue, target_queue, priority_queue, None, telemetry, lock,
   shutdown, prepare=..., make_target_request=..., gamma=..., lr_init=..., lr_min=..., target_sync_every=...)`.
6. Start `ReplayWriter`, `PriorityUpdater`, `ReplayReader`, `QTargetWorker(q_target, target_queue, ...)` and
   `learner.run`, each with `ThreadHandle`.
7. The actor steps the envs and puts one batch per step on `write_queue`.
8. Stop: `shutdown.set()`, `SHUTDOWN_SENTINEL` on each queue, join reader, learner, target worker, writer, updater.

Prioritized replay with a value network, actors and checkpoints:

1. Create the buffer, telemetry, store, and queues: `write_queue`, bounded `prefetch_queue`, `priority_queue`,
   `stats_q`, `request_queue`, checkpoint `write_queue`.
2. Start `StatsProcessor` and `CheckpointWriter`, then `ReplayWriter`, `PriorityUpdater`, `ReplayReader`, then the
   `QNetworkWorker`s, then the actors and the learner.
3. The learner takes `(indices, is_weights)` from `prefetch_queue`, gathers the rows from the buffer, trains, puts
   `(indices, |td|)` on `priority_queue`, and puts `CheckpointRequest(update, cpu_state_dict, dims)` on the checkpoint
   queue.
4. Stop: `shutdown.set()`, put `ai_runtime.SHUTDOWN_SENTINEL` on each queue, join producers before consumers
   (actors, reader, learner, then writer, updater, checkpoint writer, stats). `ReplayWriter` and `CheckpointWriter`
   drain their queues before exiting.

## Rules that are easy to break

- PPO is on-policy: without `PPOTrainer`, call `updater.wait_idle()` after `submit` and before collecting the next
  rollout. With it, start no other user of the worker and leave the advantages to the trainer.
- Collect with the model in eval mode and under `torch.no_grad()`; the stored log-probs must match what the update
  recomputes.
- Use `PPOPolicyNetwork.masked_log_probs(logits, invalid)` on both sides; `invalid` is True for a masked action.
  A numpy rollout uses `PPOPolicyNetwork.masked_log_probs_np`, which applies the same `MASKED_LOGIT` floor.
- `PPORolloutBatch.rows` is an episode id. Give an env a new id when its episode ends and when the rollout ends, so
  GAE never joins two segments.
- `forward` returns `(B, A + 1)`; split with `PPOPolicyNetwork.split(out)` into logits and value.
- The buffer emits `DrawEvent` / `OverwriteEvent` with `put_nowait` and drops them on a full `stats_q`; the
  `CheckpointEvent` from `CheckpointWriter` is never dropped.
- `CheckpointRequest.state_dict` is a CPU copy taken by the learner; the writer thread only writes it.
- `ReplayReader` waits until `len(buffer) >= max(min_buffer_size, batch_size)` before the first draw.
- Actor and learner share one process and the GIL. Pace the actor to the learner (wait while it is more than a
  set number of transitions ahead), or the actor's Python loop starves the learner thread.
- With small networks on the CPU, set `os.environ["OMP_NUM_THREADS"] = "1"` before importing torch. Every thread
  that calls torch gets its own OpenMP pool, and `torch.set_num_threads` does not reach threads started later.
- `QTargetWorker` is the only code that forwards `q_target` or loads weights into it. Send it a cloned state dict,
  taken under the learner's lock.
- `DQNLearner` calls `make_target_request` before `prepare`, so the target thread works while the learner prepares
  and runs its own forwards. `release` runs after every batch, also when the step raised.
- `decode(request)` runs on the target thread; put input preparation there (unpacking, host-to-device copies) so
  the learner's thread does not pay for it.
- Store a truncated (time-limit) step with `done=False`; only a terminal step ends the bootstrap.
- `codec.encode` must return every column the codec names; `push_batch` raises otherwise.

## Examples

- `examples/cartpole_ppo/train.py` — PPO on gymnasium CartPole, CPU, seeded. `pip install "rl-core[examples]"`,
  then `python examples/cartpole_ppo/train.py`.
- `examples/cartpole_dqn/train.py` — Double DQN with prioritized replay on CartPole, CPU: `DenseCodec`, the
  writer, reader, priority-updater and target-network threads, `DQNLearner`.
  `python examples/cartpole_dqn/train.py`.

## The five repos

| Repo | Gives | Depends on |
|---|---|---|
| [ai-runtime](https://github.com/ekazum/ai-runtime) | threads, `SHUTDOWN_SENTINEL`, `ThreadStages`, GPU batching, service supervision, SQLite stores, checkpoint managers, `YamlConfig`, logging | standard library; `[torch]`, `[config]` |
| [ai-llm](https://github.com/ekazum/ai-llm) | an LLM generation service: engine, GPU worker, server, client, stats store | ai-runtime |
| [ai-ui](https://github.com/ekazum/ai-ui) | a Flask web shell and a runs dashboard over a SQLite runs DB | ai-runtime |
| rl-core | the RL threads, the replay buffer and the data types above | ai-runtime |
| [ai-pipeline](https://github.com/ekazum/ai-pipeline) | data preparation stages | ai-runtime; `[embed]`, `[llm]`, `[reduce]` |
