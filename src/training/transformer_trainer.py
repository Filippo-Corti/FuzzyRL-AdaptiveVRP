from __future__ import annotations

import time
import torch
from pathlib import Path

from ..agent.transformer_agent import TransformerAgent
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
        self._adv_ema: float | None = None
        self._adv_ema_alpha: float = 0.05  # same smoothing as baseline

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
                node_features, truck_state, mask = self.env.get_state()
                actions, _ = self.agent.select_action(
                    node_features, truck_state, mask, greedy=True
                )
                baseline += self.env.step(actions)
        self._baseline_rewards = baseline

        # --- Reset for sampled rollout ---
        self.env.reset()
        self._sampled_rewards = torch.zeros(B, device=d)
        self._log_probs_list = []

    def step(self) -> dict:
        """
        Execute one env step of the sampled rollout.

        Should only be called after reset_episode() and while not all_done().

        Returns a metrics dict with current running totals.
        """
        node_features, truck_state, mask = self.env.get_state()
        actions, log_probs = self.agent.select_action(
            node_features, truck_state, mask, greedy=False
        )
        rewards = self.env.step(actions)

        assert self._sampled_rewards is not None
        assert self._baseline_rewards is not None
        self._sampled_rewards += rewards
        self._log_probs_list.append(log_probs)

        return {
            "sampled_reward_mean": self._sampled_rewards.mean().item(),
            "baseline_reward_mean": self._baseline_rewards.mean().item(),
            "advantage_mean": (self._sampled_rewards - self._baseline_rewards)
            .mean()
            .item(),
            "steps_taken": len(self._log_probs_list),
        }

    def update(self) -> float:
        """
        Run the REINFORCE gradient update for the completed episode.

        Should only be called after all_done() is True.
        Returns the scalar loss value.
        """
        assert self._sampled_rewards is not None
        assert self._baseline_rewards is not None
        advantage = self._sampled_rewards - self._baseline_rewards  # (B,)
        log_probs_total = torch.stack(self._log_probs_list, dim=0).sum(dim=0)  # (B,)
        loss = -(advantage.detach() * log_probs_total).mean()

        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.encoder.parameters())
            + list(self.agent.decoder.parameters()),
            max_norm=1.0,
        )
        self.agent.optimizer.step()

        self._episode += 1
        adv_mean = advantage.mean().item()
        if self._adv_ema is None:
            self._adv_ema = adv_mean
        else:
            self._adv_ema = (
                1 - self._adv_ema_alpha
            ) * self._adv_ema + self._adv_ema_alpha * adv_mean
        return loss.item()

    def is_done(self) -> bool:
        """True when the current sampled rollout episode is complete."""
        return self.env.all_done()

    # ------------------------------------------------------------------
    # Full-loop interface
    # ------------------------------------------------------------------

    def train(self, num_episodes: int) -> None:
        """
        Run the full training loop for num_episodes episodes.
        Handles logging and checkpointing.
        """
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Training: {num_episodes} episodes, "
            f"batch={self.env.batch_size}, nodes={self.env.num_nodes}"
        )

        for ep in range(1, num_episodes + 1):
            t0 = time.time()

            self.reset_episode()
            # Run for a fixed upper bound — for 15 nodes, ~60 steps is always enough
            # (worst case: every other visit is a depot return)
            for _ in range(self.env.num_nodes * 4):
                self.step()
            loss = self.update()

            elapsed = time.time() - t0

            if ep % 15 == 0:
                assert self._sampled_rewards is not None
                assert self._baseline_rewards is not None
                print(
                    f"Episode {ep:5d} | "
                    f"sampled={self._sampled_rewards.mean():.3f} | "
                    f"baseline={self._baseline_rewards.mean():.3f} | "
                    f"adv={(self._sampled_rewards - self._baseline_rewards).mean():.3f} | "
                    f"adv_ema={self._adv_ema:.3f} | "
                    f"loss={loss:.4f} | "
                    f"{elapsed:.2f}s"
                )

            if ep % self.save_every == 0:
                self.save()
                print(f"Saved checkpoint → {self.save_path}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        p = Path(path) if path else self.save_path
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
        )

        ckpt = torch.load(path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])

        trainer = cls(agent, env, save_path=save_path, save_every=save_every)
        trainer._episode = ckpt.get("episode", 0)
        return trainer
