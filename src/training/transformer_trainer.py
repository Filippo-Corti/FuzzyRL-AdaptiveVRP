from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Callable, Literal, cast

import config
from ..agent.base import AgentObservation
from ..agent.transformer.agent import TransformerAgent
from ..env.batch_env import BatchVRPEnv
from .base import BaseTrainer


class TransformerTrainer(BaseTrainer):
    """
    Encapsulates the REINFORCE training loop for the TransformerAgent.

    Greedy baseline rollout is computed once at the start of each episode
    (inside reset_episode) and stored. The sampled rollout is then driven
    step-by-step via step(), and the gradient update is triggered by update().

    train() is a convenience wrapper that calls all three in the right order.
    """

    def __init__(
        self,
        agent: TransformerAgent,
        env: BatchVRPEnv,
        save_path: str = "checkpoints/transformer.pt",
        save_every: int = 500,
    ):
        self.agent = agent
        self.env = env
        self.save_path = Path(save_path)
        self.save_every = save_every

        # Per-episode accumulators (sampled rollout)
        self._sampled_rewards: torch.Tensor | None = None
        self._log_probs_list: list[torch.Tensor] = []

        # Stored from greedy rollout
        self._baseline_rewards: torch.Tensor | None = None

        self._episode: int = 0
        self._baseline_ema: float | None = None
        self._baseline_ema_alpha: float = 0.05
        self._adv_ema: float | None = None
        self._adv_ema_alpha: float = 0.05
        self._relative_gap_ema: float | None = None
        self._relative_gap_ema_alpha: float = 0.05

    def _assert_episode_ready(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._sampled_rewards is not None
        assert self._baseline_rewards is not None
        return self._sampled_rewards, self._baseline_rewards

    def _checkpoint_path_for_episode(self, episode: int) -> Path:
        return self.save_path.with_name(
            f"{self.save_path.stem}-{episode}{self.save_path.suffix}"
        )

    @property
    def episode(self) -> int:
        return self._episode

    # ------------------------------------------------------------------
    # Step-by-step interface
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        """
        Start a new episode.

        Runs the full greedy baseline rollout first (no gradients),
        then resets the env for the sampled rollout.
        """
        B = self.env.batch_size
        d = self.agent.device

        # --- Greedy baseline ---
        self.env.reset()
        baseline = torch.zeros(B, device=d)
        with torch.no_grad():
            while not self.env.all_done():
                env_obs = self.env.get_state()
                decision = self.agent.select_action(
                    AgentObservation(
                        node_features=env_obs.node_features,
                        truck_state=env_obs.truck_state,
                        mask=env_obs.mask,
                    ),
                    greedy=True,
                )
                baseline += self.env.step(decision.actions)
        self._baseline_rewards = baseline

        # --- Reset for sampled rollout ---
        self.env.reset()
        self._sampled_rewards = torch.zeros(B, device=d)
        self._log_probs_list = []

    def step(self) -> None:
        """
        Execute one env step of the sampled rollout.

        Should only be called after reset_episode() and while not all_done().

        Returns a metrics dict with current running totals.
        """
        env_obs = self.env.get_state()
        decision = self.agent.select_action(
            AgentObservation(
                node_features=env_obs.node_features,
                truck_state=env_obs.truck_state,
                mask=env_obs.mask,
            ),
            greedy=False,
        )
        rewards = self.env.step(decision.actions)

        sampled_rewards, _ = self._assert_episode_ready()
        sampled_rewards += rewards
        self._log_probs_list.append(decision.log_probs)

    def update(self) -> float:
        """
        Run the REINFORCE gradient update for the completed episode.

        Should only be called after all_done() is True.
        Returns the scalar loss value.
        """
        sampled_rewards, baseline_rewards = self._assert_episode_ready()
        advantage = sampled_rewards - baseline_rewards  # (B,)
        log_probs_total = torch.stack(self._log_probs_list, dim=0).sum(dim=0)  # (B,)
        loss = -(advantage.detach() * log_probs_total).mean()

        self.agent.optimizer.zero_grad()
        loss.backward()
        # Clip global gradient norm to avoid unstable policy updates and NaN logits.
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.encoder.parameters())
            + list(self.agent.decoder.parameters()),
            max_norm=1.0,
        )
        self.agent.optimizer.step()

        self._episode += 1
        baseline_mean = baseline_rewards.mean().item()
        if self._baseline_ema is None:
            self._baseline_ema = baseline_mean
        else:
            self._baseline_ema = (
                1 - self._baseline_ema_alpha
            ) * self._baseline_ema + self._baseline_ema_alpha * baseline_mean

        adv_mean = advantage.mean().item()
        if self._adv_ema is None:
            self._adv_ema = adv_mean
        else:
            self._adv_ema = (
                1 - self._adv_ema_alpha
            ) * self._adv_ema + self._adv_ema_alpha * adv_mean

        denom = max(abs(baseline_mean), 1e-9)
        relative_gap = (sampled_rewards.mean().item() - baseline_mean) / denom
        if self._relative_gap_ema is None:
            self._relative_gap_ema = relative_gap
        else:
            self._relative_gap_ema = (
                1 - self._relative_gap_ema_alpha
            ) * self._relative_gap_ema + self._relative_gap_ema_alpha * relative_gap
        return loss.item()

    def is_done(self) -> bool:
        """True when the current sampled rollout episode is complete."""
        return self.env.all_done()

    # ------------------------------------------------------------------
    # Full-loop interface
    # ------------------------------------------------------------------

    def train(
        self,
        num_episodes: int,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        """
        Run the full training loop for num_episodes episodes.
        Handles logging and checkpointing.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Training: {num_episodes} episodes from episode {self._episode}, "
            f"batch={self.env.batch_size}, nodes={self.env.num_nodes}"
        )

        for _ in range(num_episodes):
            t0 = time.time()

            self.reset_episode()
            # Run for a fixed upper bound — for 15 nodes, ~60 steps is always enough
            # (worst case: every other visit is a depot return)
            for _ in range(self.env.num_nodes * 4):
                self.step()
            loss = self.update()
            current_episode = self._episode
            if progress_callback is not None:
                progress_callback(
                    {
                        "episode": current_episode,
                        "baseline": self._baseline_ema,
                        "relative_gap": self._relative_gap_ema,
                    }
                )

            elapsed = time.time() - t0

            if current_episode % 25 == 0:
                assert self._sampled_rewards is not None
                assert self._baseline_rewards is not None
                print(
                    f"Episode {current_episode:5d} | "
                    f"sampled={self._sampled_rewards.mean():.3f} | "
                    f"baseline={self._baseline_rewards.mean():.3f} | "
                    f"baseline_ema={self._baseline_ema:.3f} | "
                    f"adv={(self._sampled_rewards - self._baseline_rewards).mean():.3f} | "
                    f"adv_ema={self._adv_ema:.3f} | "
                    f"loss={loss:.4f} | "
                    f"{elapsed:.2f}s"
                )

            if current_episode % self.save_every == 0:
                self.save()
                print(
                    "Saved checkpoint → "
                    f"{self._checkpoint_path_for_episode(current_episode)}"
                )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        p = Path(path) if path else self._checkpoint_path_for_episode(self._episode)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "episode": self._episode,
                "encoder": self.agent.encoder.state_dict(),
                "decoder": self.agent.decoder.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
            },
            p,
        )

    @classmethod
    def load(
        cls,
        path: str,
        **kwargs: object,
    ) -> "TransformerTrainer":
        """Reconstruct a Trainer from a checkpoint file."""

        d_model_obj = kwargs["d_model"]
        batch_size_obj = kwargs["batch_size"]
        num_nodes_obj = kwargs["num_nodes"]
        save_path_obj = kwargs["save_path"]
        save_every_obj = kwargs["save_every"]

        assert isinstance(d_model_obj, int)
        assert isinstance(batch_size_obj, int)
        assert isinstance(num_nodes_obj, int)
        assert isinstance(save_path_obj, str)
        assert isinstance(save_every_obj, int)

        d_model = d_model_obj
        batch_size = batch_size_obj
        num_nodes = num_nodes_obj
        save_path = save_path_obj
        save_every = save_every_obj
        device = kwargs.get("device")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            assert isinstance(device, torch.device)

        agent = TransformerAgent(
            node_features=4, state_features=3, d_model=d_model, device=device
        )
        env = BatchVRPEnv(
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=device,
            depot_mode=cast(Literal["center", "random"], config.ENV_DEPOT_MODE),
            node_xy_range=config.ENV_NODE_XY_RANGE,
            demand_range=config.ENV_DEMAND_RANGE,
            capacity_range=config.ENV_CAPACITY_RANGE,
        )

        ckpt = torch.load(path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])

        trainer = cls(agent, env, save_path=save_path, save_every=save_every)
        trainer._episode = ckpt.get("episode", 0)
        return trainer
