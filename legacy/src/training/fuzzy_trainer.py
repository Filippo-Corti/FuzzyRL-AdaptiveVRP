from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Callable, Literal, cast

import config
from ..agent.base import AgentObservation
from ..agent.fuzzy.agent import FuzzyAgent
from ..env.batch_env import BatchVRPEnv
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

        self._observation: AgentObservation | None = None

        self._reward_ema: float | None = None
        self._ema_alpha: float = 0.05

        # Counts for the current episode only — reset in reset_episode()
        self._action_counts: dict[str, int] = self._fresh_action_counts()

        # Cumulative counts across all episodes — for overall distribution logging
        self._action_counts_total: dict[str, int] = self._fresh_action_counts()

        # Rolling window of per-episode distributions for stable UI display
        self._action_dist_window: list[dict[str, float]] = []
        self._action_dist_window_size: int = 50

    @staticmethod
    def _fresh_action_counts() -> dict[str, int]:
        return {"nearest": 0, "isolated": 0, "detour": 0}

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
        self._action_counts = self._fresh_action_counts()
        self._observation = self.env.get_state()

    def step(self) -> None:
        assert self._observation is not None
        observation = self._observation

        decision = self.agent.select_action(
            AgentObservation(
                node_features=observation.node_features,
                truck_state=observation.truck_state,
                mask=observation.mask,
            ),
            greedy=False,
        )

        chosen = self.agent.last_fuzzy_action
        if chosen == 0:
            self._action_counts["nearest"] += 1
            self._action_counts_total["nearest"] += 1
        elif chosen == 1:
            self._action_counts["isolated"] += 1
            self._action_counts_total["isolated"] += 1
        elif chosen == 2:
            self._action_counts["detour"] += 1
            self._action_counts_total["detour"] += 1

        reward = self.env.step(decision.actions)
        r = reward[0].item()
        self._total_reward += r
        self._steps += 1

        done = self.env.all_done()
        next_obs = self.env.get_state()

        self.agent.q_update(
            r,
            next_obs.node_features,
            next_obs.truck_state,
            next_obs.mask,
            done,
        )

        self._observation = next_obs

    def update(self) -> float:
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
        progress_callback: Callable[[dict[str, object]], None] | None = None,
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
                # Add this episode's distribution to the rolling window
                total_actions = sum(self._action_counts.values())
                if total_actions > 0:
                    episode_dist = {
                        name: count / total_actions
                        for name, count in self._action_counts.items()
                    }
                else:
                    episode_dist = {k: 0.0 for k in self._fresh_action_counts()}
                self._action_dist_window.append(episode_dist)
                if len(self._action_dist_window) > self._action_dist_window_size:
                    self._action_dist_window.pop(0)

                # Report the average over the window — stable across frames
                keys = list(self._fresh_action_counts().keys())
                smoothed_dist = {
                    k: sum(d[k] for d in self._action_dist_window) / len(self._action_dist_window)
                    for k in keys
                }

                progress_callback(
                    {
                        "episode": current_episode,
                        "epsilon": self.agent.epsilon,
                        "q_table_size": len(self.agent.q_table),
                        "action_distribution": smoothed_dist,
                    }
                )

            elapsed = time.time() - t0

            if current_episode % 500 == 0:
                # Use cumulative totals for the periodic print so the
                # percentages are stable and representative.
                grand_total = sum(self._action_counts_total.values()) or 1
                dist_str = "  ".join(
                    f"{k}={v / grand_total:.1%}"
                    for k, v in self._action_counts_total.items()
                )
                print(
                    f"Episode {current_episode:6d} | "
                    f"reward={reward:.3f} | "
                    f"ema={self._reward_ema:.3f} | "
                    f"epsilon={self.agent.epsilon:.3f} | "
                    f"q_states={len(self.agent.q_table)} | "
                    f"{dist_str} | "
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
            device = torch.device("cpu")
        else:
            assert isinstance(device, torch.device)

        agent = FuzzyAgent.load(path, device=device)
        env = BatchVRPEnv(
            batch_size=1,
            num_nodes=num_nodes,
            device=device,
            depot_mode=cast(Literal["center", "random"], config.ENV_DEPOT_MODE),
            node_xy_range=config.ENV_NODE_XY_RANGE,
            demand_range=config.ENV_DEMAND_RANGE,
            capacity_range=config.ENV_CAPACITY_RANGE,
        )
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