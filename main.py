"""main.py - project entrypoint."""

from __future__ import annotations

from typing import Literal, cast

import config
from src.viz.app import AppConfig, SimulationApp


def run(num_nodes: int = config.NUM_NODES) -> None:
    if config.AGENT_MODE not in ("transformer", "fuzzy"):
        raise ValueError("config.AGENT_MODE must be 'transformer' or 'fuzzy'")

    if config.AGENT_MODE == "transformer":
        checkpoint_path = config.CHECKPOINT_TRANSFORMER_PATH
    else:
        checkpoint_path = config.CHECKPOINT_FUZZY_PATH

    agent_mode = cast(Literal["transformer", "fuzzy"], config.AGENT_MODE)
    app_cfg = AppConfig(
        agent_mode=agent_mode,
        checkpoint_path=checkpoint_path,
        poll_interval_s=config.POLL_INTERVAL_S,
        default_speed=config.DEFAULT_SPEED,
        speed_step=config.SPEED_STEP,
        speed_min=config.SPEED_MIN,
        speed_max=config.SPEED_MAX,
    )
    SimulationApp(app_cfg, num_nodes=num_nodes).run()


if __name__ == "__main__":
    run()
