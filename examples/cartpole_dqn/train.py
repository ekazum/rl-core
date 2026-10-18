"""Double DQN with prioritized replay on CartPole, from rl-core's parts.

    pip install "rl-core[examples]"
    python examples/cartpole_dqn/train.py

rl-core supplies the buffer, its three client threads (writer, reader,
priority updater), the target-network thread and the learner thread; this
file writes the network, the env loop, the batch type, the telemetry counter
and the learner's two hooks, and prints the mean return.
"""

import argparse
import logging
import os
import queue
import threading
from concurrent.futures import Future
from typing import NamedTuple

# One OpenMP thread per torch-calling thread; set before torch starts its pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np
import torch
from ai_runtime.pipeline import SHUTDOWN_SENTINEL
from ai_runtime.thread_handle import ThreadHandle
from torch import nn

from rl_core import (DenseCodec, DQNBatch, DQNLearner, PrioritizedReplayBuffer,
                     PriorityUpdater, QTargetWorker, ReplayReader, ReplayWriter, TargetRequest)

OBS_DIM, N_ACTIONS = 4, 2


class StepBatch(NamedTuple):
    """One env step of every env: the rows the buffer stores."""

    obs: np.ndarray        # (N, 4) float32
    action: np.ndarray     # (N,)   int64
    reward: np.ndarray     # (N,)   float32
    next_obs: np.ndarray   # (N, 4) float32
    done: np.ndarray       # (N,)   bool — terminated, not truncated

    @property
    def num_rows(self) -> int:
        return self.obs.shape[0]


CODEC = DenseCodec({
    "obs": ((OBS_DIM,), np.float32),
    "action": ((), np.int64),
    "reward": ((), np.float32),
    "next_obs": ((OBS_DIM,), np.float32),
    "done": ((), np.bool_),
})


class Counter:
    """The telemetry the replay threads and the learner report to; counts only."""

    def __init__(self) -> None:
        self.transitions_pushed = 0
        self.updates = 0

    def note_transitions(self, n: int) -> None:
        self.transitions_pushed += n

    def log_buffer_stats(self, episode_index: int, metrics: dict) -> None:
        pass

    def note_update(self, update_count: int) -> None:
        self.updates = update_count


def q_network() -> nn.Module:
    return nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, N_ACTIONS))


def learner_hooks(buffer: PrioritizedReplayBuffer):
    """The learner's two hooks; a queued item is the reader's ``(indices, is_weights)``."""

    def make_target_request(item, future: Future) -> TargetRequest:
        indices, _ = item
        next_obs = np.empty((indices.shape[0], OBS_DIM), np.float32)
        buffer.gather_into(indices, {"next_obs": next_obs})
        return TargetRequest(torch.from_numpy(next_obs), future)

    def prepare(item) -> DQNBatch:
        indices, weights = item
        rows = {k: torch.from_numpy(v) for k, v in buffer.gather(indices).items()}
        return DQNBatch(states=rows["obs"], next_states=rows["next_obs"],
                        actions=rows["action"], rewards=rows["reward"], dones=rows["done"],
                        is_weights=torch.from_numpy(weights), indices=indices)

    return prepare, make_target_request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-steps", type=int, default=12_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    torch.manual_seed(args.seed)
    # Small networks: one torch thread each, so the actor and learner don't contend.
    torch.set_num_threads(1)

    shutdown = threading.Event()
    buffer = PrioritizedReplayBuffer(
        capacity=50_000, codec=CODEC, alpha=0.6, beta_is_start=0.4, n_anneal=args.env_steps,
        priority_floor=1e-3, seed_seq=np.random.SeedSequence(args.seed))
    counter = Counter()
    write_queue, priority_queue, target_queue = queue.Queue(), queue.Queue(), queue.Queue()
    prefetch_queue = queue.Queue(maxsize=8)
    q_online, network_lock = q_network(), threading.Lock()
    q_target = q_network()
    q_target.load_state_dict(q_online.state_dict())
    prepare, make_target_request = learner_hooks(buffer)
    # No write_queue: this example keeps no checkpoints.
    learner = DQNLearner(
        q_online, torch.optim.Adam(q_online.parameters(), lr=args.lr), prefetch_queue,
        target_queue, priority_queue, None, counter, network_lock, shutdown,
        prepare=prepare, make_target_request=make_target_request, gamma=args.gamma,
        lr_init=args.lr, lr_min=args.lr, target_sync_every=args.target_sync,
        push_per_update_ratio=args.train_every)

    threads = {
        "writer": ThreadHandle(ReplayWriter(buffer, write_queue, counter, shutdown).run,
                               name="writer"),
        "updater": ThreadHandle(PriorityUpdater(buffer, priority_queue, shutdown).run,
                                name="updater"),
        "reader": ThreadHandle(ReplayReader(buffer, prefetch_queue, args.batch_size,
                                            min_buffer_size=1_000, block_batches=4,
                                            shutdown_event=shutdown).run, name="reader"),
        "target": ThreadHandle(QTargetWorker(q_target, target_queue, torch.device("cpu"),
                                             shutdown).run, name="target"),
        "learner": ThreadHandle(learner.run, name="learner"),
    }
    for t in threads.values():
        t.start()

    # The actor: ε-greedy steps of every env, one StepBatch per step onto the write queue.
    envs = [gym.make("CartPole-v1") for _ in range(args.envs)]
    obs = np.stack([env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)])
    rng = np.random.default_rng(args.seed)
    returns, finished = np.zeros(args.envs), []
    steps_per_env = args.env_steps // args.envs
    for step in range(1, steps_per_env + 1):
        eps = max(0.05, 1.0 - step / (0.4 * steps_per_env))
        with torch.no_grad(), network_lock:
            greedy = q_online(torch.from_numpy(obs.astype(np.float32))).argmax(1).numpy()
        actions = np.where(rng.random(args.envs) < eps,
                           rng.integers(N_ACTIONS, size=args.envs), greedy)
        next_obs, rewards, dones, starts = [], [], [], []
        for i, env in enumerate(envs):
            o, r, terminated, truncated, _ = env.step(int(actions[i]))
            returns[i] += r
            next_obs.append(o)
            rewards.append(r)
            dones.append(terminated)
            if terminated or truncated:
                finished.append(returns[i])
                returns[i] = 0.0
                o, _ = env.reset()
            starts.append(o)
        # A truncated row keeps done=False: its next_obs is the real last state.
        write_queue.put(StepBatch(obs.astype(np.float32), actions.astype(np.int64),
                                  np.asarray(rewards, np.float32),
                                  np.stack(next_obs).astype(np.float32), np.asarray(dones)))
        obs = np.stack(starts)
        # Pace the actor to the learner: the actor's Python loop would otherwise hold the
        # GIL and starve the learner thread.
        while (counter.transitions_pushed - counter.updates * args.train_every > 1_000
               and not shutdown.is_set()):
            shutdown.wait(0.001)
        if step % (steps_per_env // 10) == 0:
            recent = finished[-20:]
            print(f"env steps {step * args.envs:6d}: mean return "
                  f"{np.mean(recent) if recent else 0.0:6.1f} over the last {len(recent)} "
                  f"episodes, {counter.updates} updates, buffer {len(buffer)}")

    # Stop: producers first, then the consumers; the writer and updater drain.
    shutdown.set()
    for q in (write_queue, prefetch_queue, priority_queue, target_queue):
        try:
            q.put_nowait(SHUTDOWN_SENTINEL)
        except queue.Full:
            pass
    for name in ("reader", "learner", "target", "writer", "updater"):
        threads[name].join(5)


if __name__ == "__main__":
    main()
