from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src import config
from src.agents import TransformerAgent
from src.agents.fuzzy import FuzzyAgent
from src.train.fuzzy_trainer import FuzzyTrainer
from src.train.transformer_trainer import TransformerTrainer


TRAINER_EPISODES = 5000
TRANSFORMER_BATCH_SIZE = 256
FUZZY_BATCH_SIZE = 512
TRAINER_NUM_NODES = config.NUM_NODES
TRAINER_LATENESS_ALPHA = 0.2
TRAINER_GRAD_CLIP_NORM = 1.0
TRAINER_SAVE_EVERY = 25
TRAINER_TORCH_THREADS = 1
TRAINER_SEED: int | None = None


def _prepare_metrics_csv(checkpoint_path: Path) -> Path:
    csv_path = checkpoint_path.with_name(f"{checkpoint_path.stem}-metrics.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "advantage_mean",
                    "baseline_cost_mean",
                    "sampled_cost_mean",
                    "entropy_mean",
                    "tonn_cost_mean",
                    "sampled_minus_tonn",
                ]
            )

    return csv_path


def _log_episode_metrics(csv_path: Path, metrics: dict[str, float]) -> None:
    tonn_cost_mean = metrics.get("tonn_cost_mean", "")
    sampled_minus_tonn : float | str | None = metrics.get("sampled_minus_tonn")
    if sampled_minus_tonn is None:
        sampled_minus_tonn = metrics.get("sampled_minus_tonn_mean", "")

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                int(metrics["episode"]),
                float(metrics["advantage_mean"]),
                float(metrics["baseline_cost_mean"]),
                float(metrics["sampled_cost_mean"]),
                float(metrics["entropy_mean"]),
                tonn_cost_mean,
                sampled_minus_tonn,
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fuzzy or Transformer agent.")
    parser.add_argument(
        "--agent",
        choices=["fuzzy", "transformer"],
        required=True,
        help="Agent type to train.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAINER_EPISODES,
        help="Number of training episodes.",
    )
    return parser.parse_args()


def train_fuzzy(episodes: int, device: torch.device) -> None:
    checkpoint_path = Path(config.CHECKPOINT_FUZZY_PATH)
    metrics_csv_path = _prepare_metrics_csv(checkpoint_path)
    if checkpoint_path.exists():
        trainer = FuzzyTrainer.load_checkpoint(
            path=checkpoint_path,
            device=device,
            lateness_penalty_alpha=TRAINER_LATENESS_ALPHA,
            grad_clip_norm=TRAINER_GRAD_CLIP_NORM,
            checkpoint_path=str(checkpoint_path),
        )
        print(f"Loaded fuzzy checkpoint from {checkpoint_path}")
    else:
        agent = FuzzyAgent(device=device)
        trainer = FuzzyTrainer(
            agent=agent,
            device=device,
            lateness_penalty_alpha=TRAINER_LATENESS_ALPHA,
            optimizer_lr=config.FUZZY_LR,
            grad_clip_norm=TRAINER_GRAD_CLIP_NORM,
            checkpoint_path=str(checkpoint_path),
        )
        print("Initialised new Fuzzy trainer")

    print(
        f"Training Fuzzy for {episodes} episodes "
        f"(batch={FUZZY_BATCH_SIZE}, nodes={TRAINER_NUM_NODES}, device={device})"
    )

    for _ in range(episodes):
        should_compare = (trainer.episode + 1) % int(TRAINER_SAVE_EVERY) == 0
        metrics = trainer.train_episode(
            batch_size=FUZZY_BATCH_SIZE,
            num_nodes=TRAINER_NUM_NODES,
            compare_with_tonn=should_compare,
        )

        _log_episode_metrics(metrics_csv_path, metrics)

        print(
            f"Episode {int(metrics['episode']):5d} | "
            f"adv={metrics['advantage_mean']:.4f} | "
            f"entropy={metrics['entropy_mean']:.4f} | "
            f"loss={metrics['loss']:.4f} | "
            f"baseline={metrics['baseline_cost_mean']:.4f} | "
            f"sampled={metrics['sampled_cost_mean']:.4f}"
        )

        if "tonn_cost_mean" in metrics:
            print(
                f"TONN monitor | tonn={metrics['tonn_cost_mean']:.4f} | "
                f"sampled-tonn={metrics['sampled_minus_tonn']:.4f}"
            )

        if int(metrics["episode"]) % int(TRAINER_SAVE_EVERY) == 0:
            ckpt = checkpoint_path.with_name(
                f"{checkpoint_path.stem}-{int(metrics['episode'])}{checkpoint_path.suffix}"
            )
            trainer.save_checkpoint(ckpt)
            print(f"Saved checkpoint -> {ckpt}")

    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved final checkpoint -> {checkpoint_path}")


def train_transformer(episodes: int, device: torch.device) -> None:
    checkpoint_path = Path(config.CHECKPOINT_TRANSFORMER_PATH)
    metrics_csv_path = _prepare_metrics_csv(checkpoint_path)
    if checkpoint_path.exists():
        trainer = TransformerTrainer.load_checkpoint(
            path=checkpoint_path,
            device=device,
            lateness_penalty_alpha=TRAINER_LATENESS_ALPHA,
            grad_clip_norm=TRAINER_GRAD_CLIP_NORM,
            checkpoint_path=str(checkpoint_path),
        )
        print(f"Loaded transformer checkpoint from {checkpoint_path}")
    else:
        agent = TransformerAgent(
            node_features=config.TRANSFORMER_NODE_FEATURES,
            state_features=config.TRANSFORMER_STATE_FEATURES,
            d_model=config.TRANSFORMER_D_MODEL,
            device=device,
        )
        trainer = TransformerTrainer(
            agent=agent,
            device=device,
            lateness_penalty_alpha=TRAINER_LATENESS_ALPHA,
            optimizer_lr=config.TRANSFORMER_LR,
            grad_clip_norm=TRAINER_GRAD_CLIP_NORM,
            checkpoint_path=str(checkpoint_path),
        )
        print("Initialized new Transformer trainer")

    print(
        f"Training Transformer for {episodes} episodes "
        f"(batch={TRANSFORMER_BATCH_SIZE}, nodes={TRAINER_NUM_NODES}, device={device})"
    )

    for _ in range(episodes):
        should_compare = (trainer.episode + 1) % int(TRAINER_SAVE_EVERY) == 0
        metrics = trainer.train_episode(
            batch_size=TRANSFORMER_BATCH_SIZE,
            num_nodes=TRAINER_NUM_NODES,
            compare_with_tonn=should_compare,
        )

        _log_episode_metrics(metrics_csv_path, metrics)

        print(
            f"Episode {int(metrics['episode']):5d} | "
            f"adv={metrics['advantage_mean']:.4f} | "
            f"entropy={metrics['entropy_mean']:.4f} | "
            f"loss={metrics['loss']:.4f} | "
            f"baseline={metrics['baseline_cost_mean']:.4f} | "
            f"sampled={metrics['sampled_cost_mean']:.4f}"
        )

        if "tonn_cost_mean" in metrics:
            print(
                f"TONN monitor | tonn={metrics['tonn_cost_mean']:.4f} | "
                f"sampled-tonn={metrics['sampled_minus_tonn_mean']:.4f}"
            )

        if int(metrics["episode"]) % int(TRAINER_SAVE_EVERY) == 0:
            ckpt = checkpoint_path.with_name(
                f"{checkpoint_path.stem}-{int(metrics['episode'])}{checkpoint_path.suffix}"
            )
            trainer.save_checkpoint(ckpt)
            print(f"Saved checkpoint -> {ckpt}")

    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved final checkpoint -> {checkpoint_path}")


def main() -> None:
    args = parse_args()

    if TRAINER_SEED is not None:
        torch.manual_seed(TRAINER_SEED)

    torch.set_num_threads(TRAINER_TORCH_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.agent == "fuzzy":
        train_fuzzy(episodes=int(args.episodes), device=device)
    else:
        train_transformer(episodes=int(args.episodes), device=device)


if __name__ == "__main__":
    main()
