"""main.py - project entrypoint."""

from __future__ import annotations

from typing import Literal, cast

import config
from src.ui.app import AppConfig, SimulationApp


def run(num_nodes: int = config.NUM_NODES) -> None:
    assert config.AGENT_MODE in ("transformer", "fuzzy")

    if config.AGENT_MODE == "transformer":
        checkpoint_path = config.CHECKPOINT_TRANSFORMER_PATH
    else:
        checkpoint_path = config.CHECKPOINT_FUZZY_PATH

    app_cfg = AppConfig(
        agent_mode=config.AGENT_MODE,
        checkpoint_path=checkpoint_path,
        poll_interval_s=config.POLL_INTERVAL_S,
        default_speed=config.DEFAULT_SPEED,
        speed_step=config.SPEED_STEP,
        speed_min=config.SPEED_MIN,
        speed_max=config.SPEED_MAX,
        trainer_batch_size=config.TRAINER_BATCH_SIZE,
        trainer_save_every=config.TRAINER_SAVE_EVERY,
        trainer_torch_threads=config.TRAINER_TORCH_THREADS,
        seed=config.SEED,
    )
    SimulationApp(app_cfg, num_nodes=num_nodes).run()


if __name__ == "__main__":
    run()
