"""PPO on CartPole from rl-core's parts: the project writes the env loop only.

    pip install "rl-core[examples]"
    python examples/cartpole_ppo/train.py

rl-core supplies the network, the rollout batch with GAE, the update thread and
its checkpoints; this file collects rollouts and prints the mean return.
"""

import argparse
import logging
import queue
import tempfile
import threading
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_core import PPOCheckpointManager, PPOPolicyNetwork, PPORolloutBatch, PPOUpdateWorker


class CartPoleCollector:
    """Steps ``n_envs`` CartPoles with the current policy; one rollout per call."""

    def __init__(self, model: PPOPolicyNetwork, n_envs: int, steps: int, seed: int) -> None:
        self._model = model
        self._envs = [gym.make("CartPole-v1") for _ in range(n_envs)]
        self._obs = np.stack([env.reset(seed=seed + i)[0]
                              for i, env in enumerate(self._envs)]).astype(np.float32)
        self._steps = steps
        self._returns = np.zeros(n_envs)
        # Each env's current segment; a new id at every episode end and rollout end.
        self._rows = np.arange(n_envs)
        self._next_row = n_envs
        self.finished: list[float] = []

    def collect(self) -> PPORolloutBatch:
        """``steps`` steps of every env, as one batch with advantages filled in."""
        parts = []
        n_actions = self._envs[0].action_space.n
        for _ in range(self._steps):
            with torch.no_grad():
                logits, values = PPOPolicyNetwork.split(self._model(torch.from_numpy(self._obs)))
                invalid = torch.zeros_like(logits, dtype=torch.bool)   # every action valid
                log_probs = PPOPolicyNetwork.masked_log_probs(logits, invalid)
                actions = torch.distributions.Categorical(logits=log_probs).sample()
            taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_obs, rewards, dones = [], [], []
            for i, env in enumerate(self._envs):
                obs, reward, terminated, truncated, _ = env.step(int(actions[i]))
                self._returns[i] += reward
                done = terminated or truncated
                if done:
                    self.finished.append(self._returns[i])
                    self._returns[i] = 0.0
                    obs, _ = env.reset()
                next_obs.append(obs)
                rewards.append(reward)
                dones.append(done)
            parts.append(PPORolloutBatch(
                observations=self._obs.copy(),
                actions=actions.numpy().astype(np.int64),
                log_probs=taken.numpy().astype(np.float32),
                values=values.numpy().astype(np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                dones=np.asarray(dones),
                invalid=np.zeros((len(self._envs), n_actions), dtype=bool),
                rows=self._rows.copy()))
            for i, done in enumerate(dones):
                if done:
                    self._rows[i] = self._next_row
                    self._next_row += 1
            self._obs = np.stack(next_obs).astype(np.float32)
        # Segments cut by the rollout's end continue under fresh ids next time.
        self._rows = np.arange(self._next_row, self._next_row + len(self._envs))
        self._next_row += len(self._envs)
        batch = PPORolloutBatch.concatenate(parts)
        batch.compute_gae(gamma=0.99, gae_lambda=0.95)
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(args.seed)

    model = PPOPolicyNetwork(input_dim=4, num_actions=2)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    shutdown = threading.Event()
    # The update thread trains on each rollout; collecting waits for it (on-policy).
    updater = PPOUpdateWorker(model, optimizer, queue.Queue(maxsize=1), torch.device("cpu"),
                              shutdown, epochs=4, minibatch_size=256)
    updater.start()
    collector = CartPoleCollector(model, args.envs, args.steps, args.seed)

    with tempfile.TemporaryDirectory() as tmp:          # removed on exit
        checkpoints = PPOCheckpointManager(Path(tmp) / "checkpoints")
        for iteration in range(1, args.iterations + 1):
            batch = collector.collect()
            updater.submit(batch)
            updater.wait_idle()
            recent = collector.finished[-20:]
            mean_return = float(np.mean(recent)) if recent else 0.0
            print(f"iteration {iteration:3d}: mean return {mean_return:6.1f} "
                  f"over the last {len(recent)} episodes")
            checkpoints.save_snapshot(model, updater.updates, {"mean_return": mean_return})
        print(f"{len(checkpoints.list_snapshots())} snapshots written")

    shutdown.set()
    updater.join(5)


if __name__ == "__main__":
    main()
