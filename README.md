# rl-core

Reinforcement learning parts for threaded training pipelines, built on
[ai-runtime](https://github.com/ekazum/ai-runtime).

## Parts

- **PPO**: `PPOPolicyNetwork` (separate policy and value stacks, masked log-probs), `PPORolloutBatch` (flat
  transitions with GAE), `PPOUpdateWorker` (the update thread), `PPOCheckpointManager` (`latest.pt`, `best.pt`,
  per-update snapshots).
- **Prioritized replay**: `SumTree`, and the buffer's client threads `ReplayWriter`, `ReplayReader`,
  `PriorityUpdater`, `StatsProcessor`. The buffer is the project's, given as a `ReplayBufferProtocol`.
- **Inference**: `QNetworkWorker` serves `QNetRequest`s from actor threads with one no-grad forward per drain.
- **Checkpoints**: `CheckpointWriter` writes `CheckpointRequest` snapshots and `LatestRequest` states to a
  `SnapshotStoreProtocol` store, off the learner's thread; `CheckpointState` is the resume schema.
- **Protocols**: `ReplayBufferProtocol`, `ReplayTelemetryProtocol`, `SnapshotStoreProtocol`.

`AGENTS.md` lists the signatures, the assembly order and the rules to keep.

## Install

```bash
pip install "rl-core @ git+https://github.com/ekazum/rl-core.git@v0.1.1"
```

## Example

```bash
pip install "rl-core[examples] @ git+https://github.com/ekazum/rl-core.git@v0.1.1"
python examples/cartpole_ppo/train.py
```

PPO on CartPole on the CPU: in one 40-iteration run, the mean return over the last 20 episodes rose from 25.6 to
190.1.

## License

Apache-2.0
