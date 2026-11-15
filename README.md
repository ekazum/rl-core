# rl-core

Reinforcement learning parts for threaded training pipelines, built on
[ai-runtime](https://github.com/ekazum/ai-runtime).

## Parts

- **PPO**: `PPOPolicyNetwork` (separate policy and value stacks, masked log-probs in torch and numpy, Gumbel-max
  sampling), `PPORolloutBatch` (flat transitions with GAE), `PPOUpdateWorker` (the update thread), `PPOTrainer`
  (the on-policy loop: collect, advantages, update, wait; the project supplies `collect` and optional hooks),
  `PPOCheckpointManager` (`latest.pt`, `best.pt`, per-update snapshots).
- **Prioritized replay**: `PrioritizedReplayBuffer` (thread-safe, columnar, sum-tree draws, β anneal) over a
  `ColumnStore`; the project's row layout comes from a codec (`RowCodecProtocol`), or `DenseCodec` stores each
  field as-is. Its client threads are `ReplayWriter`, `ReplayReader`, `PriorityUpdater`, `StatsProcessor`.
- **Inference**: `QNetworkWorker` serves `QNetRequest`s from actor threads with one no-grad forward per drain.
  `QTargetWorker` owns the target network: it answers the learner's `TargetRequest`s, optionally on its own
  device, and loads `SyncRequest` snapshots. The networks are the project's own `nn.Module`s.
- **DQN training**: `DQNLearner` runs the Double DQN loop (optional SMDP durations), the learning-rate schedule,
  target syncs and checkpoint cadences; the project supplies the optimizer and two hooks that turn a drawn batch
  into a `DQNBatch` and a target request.
- **DDPG critic training**: `DDPGLearner` is the abstract critic loop: gradient clipping, the optimizer step, the
  soft target update, priorities, a linear learning-rate schedule, snapshots and the `max_updates` stop. The
  project subclasses it with the critic's forward and backward pass, which returns a `CriticStep`.
  `GradientExchange` averages the gradients of one learner per critic copy, for a critic on several devices.
- **Checkpoints**: `CheckpointWriter` writes `CheckpointRequest` snapshots and `LatestRequest` states to a
  `SnapshotStoreProtocol` store, off the learner's thread; `CheckpointState` is the resume schema.
- **Evaluation**: `EvalTask` and `ScoreTask` are the abstract tasks a training pipeline's manager calls; the
  project fills them in. `EvalConfig` holds the settings every evaluation pass shares, and `EvalStore` owns
  `eval.db`: one dev row per scored checkpoint, the test metrics and the best checkpoint.
- **Actor feeding**: `RowSampler` gives every row of a table once per pass in a new random order per pass, with a
  snapshot for resume; `ActorFeeder` is the thread that cuts its order into batches of row ids for the actor threads,
  with an optional row cap and a pace tied to the learner's update count.
- **Protocols**: `FeederTelemetryProtocol`, `LearnerTelemetryProtocol`, `ReplayBufferProtocol`,
  `ReplayTelemetryProtocol`, `RowCodecProtocol`, `SnapshotStoreProtocol`.

`AGENTS.md` lists the signatures, the assembly order and the rules to keep.

## Install

```bash
pip install "rl-core @ git+https://github.com/ekazum/rl-core.git@v0.9.0"
```

## Examples

```bash
pip install "rl-core[examples] @ git+https://github.com/ekazum/rl-core.git@v0.9.0"
python examples/cartpole_ppo/train.py
python examples/cartpole_dqn/train.py
```

- PPO on CartPole on the CPU: in one 40-iteration run, the mean return over the last 20 episodes rose from 25.6
  to 190.1.
- Double DQN with prioritized replay on CartPole on the CPU: in one 12000-step run, the mean return over the last
  20 episodes rose from 34.5 to a peak of 183.4 and was 97.5 at the end. The threads make runs differ from one
  another.

## License

Apache-2.0
