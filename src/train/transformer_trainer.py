from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import torch

from src import config
from src.agents import TONNAgent, TransformerAgent
from src.vrp import VRPEnvironmentBatch, VRPInstanceBatch


class TransformerTrainer:
    """Basic REINFORCE trainer with self-greedy baseline and optional TONN monitoring."""

    def __init__(
        self,
        agent: TransformerAgent,
        device: torch.device,
        lateness_penalty_alpha: float = 0.2,
        tonn_baseline: TONNAgent | None = None,
        optimizer_lr: float = config.TRANSFORMER_LR,
        grad_clip_norm: float = 1.0,
        checkpoint_path: str = config.CHECKPOINT_TRANSFORMER_PATH,
        baseline_update_freq: int = 50,
        entropy_coef: float = 0.03,
    ):
        self.agent = agent
        self.device = device
        self.lateness_penalty_alpha = float(lateness_penalty_alpha)
        self.tonn_baseline = tonn_baseline if tonn_baseline is not None else TONNAgent()
        self.optimizer_lr = float(optimizer_lr)
        self.optimizer = torch.optim.Adam(
            list(self.agent.encoder.parameters())
            + list(self.agent.decoder.parameters()),
            lr=self.optimizer_lr,
        )
        self.grad_clip_norm = float(grad_clip_norm)
        self.checkpoint_path = Path(checkpoint_path)
        self.baseline_update_freq = int(baseline_update_freq)
        self.entropy_coef = float(entropy_coef)
        self.episode = 0

        # Frozen baseline: greedy copy of current agent, updated every
        # baseline_update_freq episodes.
        self.baseline_agent = deepcopy(self.agent)
        self.baseline_agent.eval()
        for p in self.baseline_agent.parameters():
            p.requires_grad_(False)

    def _build_training_batch(
        self, batch_size: int, num_nodes: int
    ) -> VRPInstanceBatch:
        return VRPInstanceBatch(
            batch_size=batch_size,
            num_nodes=num_nodes,
            device=self.device,
            depot_mode=config.ENV_DEPOT_MODE,
            node_xy_range=config.ENV_NODE_XY_RANGE,
            weight_range=config.ENV_WEIGHT_RANGE,
            W_value=config.ENV_W_FIXED,
            initial_visible_ratio=config.ENV_INITIAL_VISIBLE_RATIO,
            window_length_range=config.ENV_WINDOW_LENGTH_RANGE,
            cluster_count_range=config.ENV_CLUSTER_COUNT_RANGE,
            outlier_count_range=config.ENV_OUTLIER_COUNT_RANGE,
            cluster_std_range=config.ENV_CLUSTER_STD_RANGE,
        )

    def _run_tonn_baseline(self, instance_batch: VRPInstanceBatch) -> torch.Tensor:
        tonn_env = VRPEnvironmentBatch(
            instance_batch=instance_batch,
            lateness_penalty_alpha=self.lateness_penalty_alpha,
        )
        with torch.no_grad():
            tonn_cost = tonn_env.solve(self.tonn_baseline.select_actions)
        return tonn_cost

    def _run_greedy_baseline(self, instance_batch: VRPInstanceBatch) -> torch.Tensor:
        """Run frozen self-baseline greedily on the same instances."""
        env = VRPEnvironmentBatch(
            instance_batch=instance_batch,
            lateness_penalty_alpha=self.lateness_penalty_alpha,
        )

        with torch.no_grad():
            max_steps = max(4 * env.num_nodes, env.num_nodes + 1)
            steps = 0
            while not bool(env.done.all().item()) and steps < max_steps:
                obs = env.get_observation()
                logits = self.baseline_agent.forward(
                    node_features=obs["node_features"],
                    truck_state=obs["truck_state"],
                    invalid_mask=obs["invalid_action_mask"],
                )
                actions = logits.argmax(dim=-1).to(
                    torch.long
                )  # Greedy action selection

                # If no valid action exists at this step, fallback to depot wait.
                valid_any = obs["valid_action_mask"].any(dim=1)
                actions = torch.where(valid_any, actions, torch.zeros_like(actions))
                env.execute(actions)
                steps += 1

        return env.total_distance + self.lateness_penalty_alpha * env.total_lateness

    def _run_sampled_episode(
        self, instance_batch: VRPInstanceBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: log_prob_sum, total_distance, combined_cost, entropy_mean
        """
        env = VRPEnvironmentBatch(
            instance_batch=instance_batch,
            lateness_penalty_alpha=self.lateness_penalty_alpha,
        )
        B = instance_batch.batch_size

        log_probs_by_step: list[torch.Tensor] = []
        entropy_by_step: list[torch.Tensor] = []
        max_steps = max(4 * env.num_nodes, env.num_nodes + 1)
        steps = 0

        while not bool(env.done.all().item()) and steps < max_steps:
            obs = env.get_observation()
            logits = self.agent.forward(
                node_features=obs["node_features"],
                truck_state=obs["truck_state"],
                invalid_mask=obs["invalid_action_mask"],
            )

            valid_any = obs["valid_action_mask"].any(dim=1)
            actions = torch.zeros(B, dtype=torch.long, device=self.device)
            log_probs = torch.zeros(B, dtype=torch.float32, device=self.device)

            if bool(valid_any.any().item()):
                dist = torch.distributions.Categorical(logits=logits[valid_any])
                sampled_actions = dist.sample()
                actions[valid_any] = sampled_actions
                log_probs[valid_any] = dist.log_prob(sampled_actions)

                probs = torch.softmax(logits[valid_any], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                entropy_by_step.append(entropy)

            log_probs_by_step.append(log_probs)
            env.execute(actions)
            steps += 1

        if not log_probs_by_step:
            raise RuntimeError("Sampled rollout produced no steps")

        log_prob_sum = torch.stack(log_probs_by_step, dim=0).sum(dim=0)
        entropy_mean = (
            torch.stack(entropy_by_step).mean()
            if entropy_by_step
            else torch.tensor(0.0, device=self.device)
        )
        combined_cost = (
            env.total_distance + self.lateness_penalty_alpha * env.total_lateness
        )
        return log_prob_sum, env.total_distance, combined_cost, entropy_mean

    def train_episode(
        self,
        batch_size: int,
        num_nodes: int,
        compare_with_tonn: bool = False,
    ) -> dict[str, float]:
        self.agent.train()

        instance_batch = self._build_training_batch(batch_size, num_nodes)
        baseline_cost = self._run_greedy_baseline(instance_batch).detach()
        log_prob_sum, sampled_distance, sampled_cost, entropy_mean = (
            self._run_sampled_episode(instance_batch)
        )

        advantage = (
            baseline_cost - sampled_cost
        )  # positive = sampled better than baseline

        loss = (
            -(advantage.detach() * log_prob_sum).mean()
            - self.entropy_coef * entropy_mean
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), max_norm=self.grad_clip_norm
        )
        self.optimizer.step()

        self.episode += 1

        metrics: dict[str, float] = {
            "episode": float(self.episode),
            "loss": float(loss.item()),
            "baseline_cost_mean": float(baseline_cost.mean().item()),
            "sampled_cost_mean": float(sampled_cost.mean().item()),
            "sampled_distance_mean": float(sampled_distance.mean().item()),
            "advantage_mean": float(advantage.mean().item()),
            "entropy_mean": float(entropy_mean.item()),
        }

        if compare_with_tonn:
            tonn_cost = self._run_tonn_baseline(instance_batch).detach()
            metrics["tonn_cost_mean"] = float(tonn_cost.mean().item())
            metrics["sampled_minus_tonn_mean"] = float(
                (sampled_cost - tonn_cost).mean().item()
            )

        if (
            self.baseline_update_freq > 0
            and self.episode % self.baseline_update_freq == 0
        ):
            self.baseline_agent.encoder.load_state_dict(self.agent.encoder.state_dict())
            self.baseline_agent.decoder.load_state_dict(self.agent.decoder.state_dict())
            self.baseline_agent.eval()

        return metrics

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "episode": self.episode,
                "optimizer_lr": self.optimizer_lr,
                "node_features": self.agent.encoder.input_proj.in_features,
                "state_features": self.agent.decoder.query_proj.in_features,
                "d_model": self.agent.encoder.input_proj.out_features,
                "encoder": self.agent.encoder.state_dict(),
                "decoder": self.agent.decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            target,
        )
        return target

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: torch.device,
        lateness_penalty_alpha: float = 0.2,
        tonn_baseline: TONNAgent | None = None,
        grad_clip_norm: float = 1.0,
        checkpoint_path: str = config.CHECKPOINT_TRANSFORMER_PATH,
    ) -> "TransformerTrainer":
        ckpt = torch.load(path, map_location=device)

        agent = TransformerAgent(
            node_features=int(ckpt["node_features"]),
            state_features=int(ckpt["state_features"]),
            d_model=int(ckpt["d_model"]),
            device=device,
        )
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])

        trainer = cls(
            agent=agent,
            device=device,
            lateness_penalty_alpha=lateness_penalty_alpha,
            tonn_baseline=tonn_baseline,
            optimizer_lr=float(ckpt.get("optimizer_lr", config.TRANSFORMER_LR)),
            grad_clip_norm=grad_clip_norm,
            checkpoint_path=checkpoint_path,
        )

        if ckpt.get("optimizer"):
            trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.episode = int(ckpt.get("episode", 0))
        return trainer
