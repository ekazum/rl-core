"""Double DQN with prioritized replay on CartPole, from rl-core's parts.

    pip install "rl-core[examples]"
    python examples/cartpole_dqn/train.py

rl-core supplies the buffer, its three client threads (writer, reader,
priority updater) and the target-network thread; this file writes the
network, the env loop, the batch type, the telemetry counter and the learner
thread, and prints the mean return.
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

from rl_core import (DenseCodec, PrioritizedReplayBuffer, PriorityUpdater, QTargetWorker,
                     ReplayReader, ReplayWriter, SyncRequest, TargetRequest)

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
    """The telemetry the replay threads report to; this example keeps the count only."""

    def __init__(self) -> None:
        self.transitions_pushed = 0

    def note_transitions(self, n: int) -> None:
        self.transitions_pushed += n

    def log_buffer_stats(self, episode_index: int, metrics: dict) -> None:
        pass


def q_network() -> nn.Module:
    return nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, N_ACTIONS))


class Learner:
    """Learner thread: prefetched draws → gather → DDQN step → new priorities.

    Target values come from the ``QTargetWorker`` thread, which owns ``q_target``;
    the learner sends it ``TargetRequest``s and, every ``target_sync`` updates, a
    ``SyncRequest``.
    """

    def __init__(self, buffer: PrioritizedReplayBuffer, q_online: nn.Module,
                 network_lock: threading.Lock, prefetch_queue: queue.Queue,
                 priority_queue: queue.Queue, target_queue: queue.Queue, counter: Counter,
                 args, shutdown: threading.Event) -> None:
        self._buffer = buffer
        self._q_online = q_online
        self._target_queue = target_queue
        self._lock = network_lock
        self._optimizer = torch.optim.Adam(q_online.parameters(), lr=args.lr)
        self._prefetch_queue = prefetch_queue
        self._priority_queue = priority_queue
        self._counter = counter
        self._args = args
        self._shutdown = shutdown
        self.updates = 0

    def run(self) -> None:
        while not self._shutdown.is_set():
            # Hold the update rate to one per `train_every` stored transitions.
            if self.updates * self._args.train_every >= self._counter.transitions_pushed:
                self._shutdown.wait(0.001)
                continue
            try:
                item = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is SHUTDOWN_SENTINEL:
                break
            indices, weights = item
            self._priority_queue.put((indices, self._step(indices, weights)))
            self.updates += 1
            if self.updates % self._args.target_sync == 0:
                with self._lock:
                    snapshot = {k: v.clone() for k, v in self._q_online.state_dict().items()}
                self._target_queue.put(SyncRequest(snapshot))

    def _step(self, indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """One Double DQN update; returns |TD| per row."""
        rows = {k: torch.from_numpy(v) for k, v in self._buffer.gather(indices).items()}
        # Ask for the target values first; the target thread works while this one does.
        target_values = Future()
        self._target_queue.put(TargetRequest(rows["next_obs"], target_values))
        with torch.no_grad():
            next_action = self._q_online(rows["next_obs"]).argmax(1, keepdim=True)
            next_q = target_values.result().gather(1, next_action).squeeze(1)
            target = rows["reward"] + self._args.gamma * next_q * (~rows["done"]).float()
        with self._lock:
            q = self._q_online(rows["obs"]).gather(1, rows["action"].unsqueeze(1)).squeeze(1)
            td = target - q
            loss = (torch.from_numpy(weights) * nn.functional.smooth_l1_loss(
                q, target, reduction="none")).mean()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
        return td.detach().abs().numpy()


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
    learner = Learner(buffer, q_online, network_lock, prefetch_queue, priority_queue,
                      target_queue, counter, args, shutdown)

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
        while (counter.transitions_pushed - learner.updates * args.train_every > 1_000
               and not shutdown.is_set()):
            shutdown.wait(0.001)
        if step % (steps_per_env // 10) == 0:
            recent = finished[-20:]
            print(f"env steps {step * args.envs:6d}: mean return "
                  f"{np.mean(recent) if recent else 0.0:6.1f} over the last {len(recent)} "
                  f"episodes, {learner.updates} updates, buffer {len(buffer)}")

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
