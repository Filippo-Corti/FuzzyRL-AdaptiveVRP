from __future__ import annotations

from pathlib import Path

import torch

import config
from src.agents.fuzzy import FuzzyAgent
from src.train.fuzzy_trainer import FuzzyTrainer


def main() -> None:
    if config.SEED is not None:
        torch.manual_seed(config.SEED)

    torch.set_num_threads(config.TRAINER_TORCH_THREADS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(config.CHECKPOINT_FUZZY_PATH)
    if checkpoint_path.exists():
        trainer = FuzzyTrainer.load_checkpoint(
            path=checkpoint_path,
            device=device,
            lateness_penalty_alpha=config.TRAINER_LATENESS_ALPHA,
            grad_clip_norm=config.TRAINER_GRAD_CLIP_NORM,
            checkpoint_path=config.CHECKPOINT_FUZZY_PATH,
        )
        print(f"Loaded fuzzy checkpoint from {checkpoint_path}")
    else:
        agent = FuzzyAgent(device=device)
        trainer = FuzzyTrainer(
            agent=agent,
            device=device,
            lateness_penalty_alpha=config.TRAINER_LATENESS_ALPHA,
            optimizer_lr=config.FUZZY_LR,
            grad_clip_norm=config.TRAINER_GRAD_CLIP_NORM,
            checkpoint_path=config.CHECKPOINT_FUZZY_PATH,
        )
        print("Initialised new Fuzzy trainer")

    print(
        f"Training FuzzyAgent for {config.TRAINER_EPISODES} episodes "
        f"(batch={config.FUZZY_BATCH_SIZE}, nodes={config.TRAINER_NUM_NODES}, device={device})"
    )

    for _ in range(config.TRAINER_EPISODES):
        should_compare = (trainer.episode + 1) % int(config.TRAINER_SAVE_EVERY) == 0
        metrics = trainer.train_episode(
            batch_size=config.FUZZY_BATCH_SIZE,
            num_nodes=config.TRAINER_NUM_NODES,
            compare_with_tonn=should_compare,
        )

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

        if int(metrics["episode"]) % int(config.TRAINER_SAVE_EVERY) == 0:
            ckpt = checkpoint_path.with_name(
                f"{checkpoint_path.stem}-{int(metrics['episode'])}{checkpoint_path.suffix}"
            )
            trainer.save_checkpoint(ckpt)
            print(f"Saved checkpoint -> {ckpt}")

    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved final checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()
