from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Callable

from ..agent.base import AgentObservation as PolicyObservation
from ..agent.fuzzy.agent import FuzzyAgent
from ..env.batch_env import AgentObservation, BatchVRPEnv
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
        self._observation: AgentObservation | None = None

        # Running average reward for logging
        self._reward_ema: float | None = None
        self._ema_alpha: float = 0.05

    @property
    def episode(self) -> int:
        return self._episode

    def _checkpoint_path_for_episode(self, episode: int) -> Path:
        return self.save_path.with_name(
            f"{self.save_path.stem}-{episode}{self.save_path.suffix}"
        )

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        self.env.reset()
        self._total_reward = 0.0
        self._steps = 0
        # Cache initial state
        self._observation = self.env.get_state()

    def step(self) -> None:
        observation = self._observation
        assert observation is not None

        # Agent selects action (stores last state/action internally)
        decision = self.agent.select_action(
            PolicyObservation(
                node_features=observation.node_features,
                truck_state=observation.truck_state,
                mask=observation.mask,
            ),
            greedy=False,
        )

        # Step env
        reward = self.env.step(decision.actions)
        r = reward[0].item()
        self._total_reward += r
        self._steps += 1

        # Get next state
        done = self.env.all_done()
        next_obs = self.env.get_state()

        # Q-update
        self.agent.q_update(
            r,
            next_obs.node_features,
            next_obs.truck_state,
            next_obs.mask,
            done,
        )

        # Advance cached state
        self._observation = next_obs

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

    def train(
        self,
        num_episodes: int,
        progress_callback: Callable[[dict[str, int | float | None]], None] | None = None,
    ) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[fuzzy] Training: {num_episodes} episodes from episode {self._episode}, "
            f"nodes={self.env.num_nodes}"
        )

        for _ in range(num_episodes):
            t0 = time.time()

            self.reset_episode()
            for _ in range(self.env.num_nodes * 4):
                if self.is_done():
                    break
                self.step()
            reward = self.update()
            current_episode = self._episode
            if progress_callback is not None:
                progress_callback({"episode": current_episode})

            elapsed = time.time() - t0

            if current_episode % 500 == 0:
                print(
                    f"Episode {current_episode:6d} | "
                    f"reward={reward:.3f} | "
                    f"ema={self._reward_ema:.3f} | "
                    f"epsilon={self.agent.epsilon:.3f} | "
                    f"q_states={len(self.agent.q_table)} | "
                    f"{elapsed:.3f}s"
                )

            if current_episode % self.save_every == 0:
                self.save()
                print(
                    "[fuzzy] Saved checkpoint → "
                    f"{self._checkpoint_path_for_episode(current_episode)}"
                )

    def save(self, path: str | None = None) -> None:
        p = Path(path) if path else self._checkpoint_path_for_episode(self._episode)
        self.agent.save(str(p))
        # Save trainer metadata alongside agent in a companion file
        torch.save(
            {"episode": self._episode, "reward_ema": self._reward_ema}, str(p) + ".meta"
        )

    @classmethod
    def load(
        cls,
        path: str,
        **kwargs: object,
    ) -> "FuzzyTrainer":
        num_nodes_obj = kwargs["num_nodes"]
        save_path_obj = kwargs["save_path"]
        save_every_obj = kwargs["save_every"]

        assert isinstance(num_nodes_obj, int)
        assert isinstance(save_path_obj, str)
        assert isinstance(save_every_obj, int)

        num_nodes = num_nodes_obj
        save_path = save_path_obj
        save_every = save_every_obj
        device = kwargs.get("device")

        if device is None:
            device = torch.device("cpu")  # fuzzy agent is CPU-only
        else:
            assert isinstance(device, torch.device)

        agent = FuzzyAgent.load(path, device=device)
        env = BatchVRPEnv(batch_size=1, num_nodes=num_nodes, device=device)
        trainer = cls(
            agent=agent,
            env=env,
            save_path=save_path,
            save_every=save_every,
        )
        meta_path = path + ".meta"
        if Path(meta_path).exists():
            meta = torch.load(meta_path)
            trainer._episode = meta.get("episode", 0)
            trainer._reward_ema = meta.get("reward_ema", None)
        return trainer
