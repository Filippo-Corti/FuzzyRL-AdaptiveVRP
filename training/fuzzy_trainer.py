from __future__ import annotations

import time
import torch
from pathlib import Path

from agent.fuzzy_agent import FuzzyAgent
from env.batch_env import BatchVRPEnv
from .base import BaseTrainer


class FuzzyTrainer(BaseTrainer):
    """
    Episode-based Q-learning trainer for FuzzyAgent.

    Runs one instance at a time (batch_size=1). Each step does a Q-update
    using the immediate reward and the next state. Epsilon decays after
    each episode.

    Matches the BaseTrainer interface so main.py can swap it in for Trainer
    with only the import changing.
    """

    def __init__(
        self,
        agent: FuzzyAgent,
        env: BatchVRPEnv,
        save_path: str = "checkpoints/fuzzy.pkl",
        save_every: int = 500,
    ):
        self.agent = agent
        self.env = env
        self.save_path = Path(save_path)
        self.save_every = save_every

        self._episode: int = 0
        self._total_reward: float = 0.0
        self._steps: int = 0

        # Store current state tensors for Q-update in step()
        self._node_features: torch.Tensor | None = None
        self._truck_state: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None

        # Running average reward for logging
        self._reward_ema: float | None = None
        self._ema_alpha: float = 0.05

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        self.env.reset()
        self._total_reward = 0.0
        self._steps = 0
        # Cache initial state
        nf, ts, mk = self.env.get_state()
        self._node_features = nf
        self._truck_state = ts
        self._mask = mk

    def step(self) -> dict:
        nf, ts, mk = self._node_features, self._truck_state, self._mask

        # Agent selects action (stores last state/action internally)
        actions, _ = self.agent.select_action(nf, ts, mk, greedy=False)

        # Step env
        reward = self.env.step(actions)
        r = reward[0].item()
        self._total_reward += r
        self._steps += 1

        # Get next state
        done = self.env.all_done()
        next_nf, next_ts, next_mk = self.env.get_state()

        # Q-update
        self.agent.q_update(r, next_nf, next_ts, next_mk, done)

        # Advance cached state
        self._node_features = next_nf
        self._truck_state = next_ts
        self._mask = next_mk

        return {
            "total_reward": self._total_reward,
            "steps": self._steps,
            "epsilon": self.agent.epsilon,
        }

    def update(self) -> float:
        """
        Called at episode end. Decays epsilon, updates reward EMA.
        Returns total episode reward (used as the scalar metric like loss).
        """
        self.agent.decay_epsilon()
        self._episode += 1

        r = self._total_reward
        if self._reward_ema is None:
            self._reward_ema = r
        else:
            self._reward_ema = (
                1 - self._ema_alpha
            ) * self._reward_ema + self._ema_alpha * r

        return r

    def is_done(self) -> bool:
        return self.env.all_done()

    def train(self, num_episodes: int) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fuzzy] Training: {num_episodes} episodes, nodes={self.env.num_nodes}")

        for ep in range(1, num_episodes + 1):
            t0 = time.time()

            self.reset_episode()
            for _ in range(self.env.num_nodes * 4):
                if self.is_done():
                    break
                self.step()
            reward = self.update()

            elapsed = time.time() - t0

            if ep % 500 == 0:
                print(
                    f"Episode {ep:6d} | "
                    f"reward={reward:.3f} | "
                    f"ema={self._reward_ema:.3f} | "
                    f"epsilon={self.agent.epsilon:.3f} | "
                    f"q_states={len(self.agent.q_table)} | "
                    f"{elapsed:.3f}s"
                )

            if ep % self.save_every == 0:
                self.save()
                print(f"[fuzzy] Saved checkpoint → {self.save_path}")

    def save(self, path: str | None = None) -> None:
        p = Path(path) if path else self.save_path
        self.agent.save(str(p))
        # Save trainer metadata alongside agent in a companion file
        torch.save(
            {"episode": self._episode, "reward_ema": self._reward_ema}, str(p) + ".meta"
        )

    @classmethod
    def load(
        cls,
        path: str,
        **kwargs,
    ) -> "FuzzyTrainer":
        if kwargs["device"] is None:
            device = torch.device("cpu")  # fuzzy agent is CPU-only
        else:
            device = kwargs["device"]

        agent = FuzzyAgent.load(path, device=device)
        env = BatchVRPEnv(batch_size=1, num_nodes=kwargs["num_nodes"], device=device)
        trainer = cls(
            agent=agent,
            env=env,
            save_path=kwargs["save_path"],
            save_every=kwargs["save_every"],
        )
        meta_path = path + ".meta"
        if Path(meta_path).exists():
            meta = torch.load(meta_path)
            trainer._episode = meta.get("episode", 0)
            trainer._reward_ema = meta.get("reward_ema", None)
        return trainer
