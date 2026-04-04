"""
main.py — parallel training + live visualization.

Layout:
  - Trainer runs in a background daemon thread, saving checkpoints periodically.
  - Visualization runs in the main thread (pygame requirement).
  - Every POLL_INTERVAL_S seconds, the viz thread checks whether the checkpoint
    file has been updated and hot-reloads the agent if so.

Controls:
  SPACE       pause / resume animation
  RIGHT       step one frame while paused
  R           new random instance (randomize seed)
  Ctrl+R      restart same instance (reset to stored seed)
  +/-         speed up / slow down animation
  ESC / Q     quit
"""

from __future__ import annotations

import os
import threading
import time
import torch
import pygame
from pathlib import Path

import config
from agent.transformer_agent import TransformerAgent
from env.batch_env import BatchVRPEnv
from training.transformer_trainer import Trainer
from visualization.visualization import Visualization
from viz.renderer import SimulationUI
from viz.hud import HUD
from viz.sprites import Sprites

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CHECKPOINT_PATH = "checkpoints/transformer.pt"
POLL_INTERVAL_S = 2.0  # how often to check for a new checkpoint
DEFAULT_SPEED = 0.04  # fraction of move completed per frame (~25 frames/move)
SPEED_STEP = 0.01
SPEED_MIN = 0.005
SPEED_MAX = 1.0

# Speed slider geometry
SLIDER_X = 20
SLIDER_Y = config.WINDOW_H - 40
SLIDER_W = 300
SLIDER_MIN_SPEED = SPEED_MIN
SLIDER_MAX_SPEED = 0.15  # cap slider at 0.15; keyboard can go higher


# ------------------------------------------------------------------
# Background trainer thread
# ------------------------------------------------------------------


def _run_trainer(checkpoint_path: str, num_nodes: int, batch_size: int = 64):
    """Runs forever in a daemon thread."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = TransformerAgent(
        node_features=4, state_features=3, d_model=128, device=device
    )
    env = BatchVRPEnv(batch_size=batch_size, num_nodes=num_nodes, device=device)
    trainer = Trainer(
        agent=agent,
        env=env,
        save_path=checkpoint_path,
        save_every=200,
    )

    # Resume from checkpoint if one already exists
    p = Path(checkpoint_path)
    if p.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        trainer._episode = ckpt.get("episode", 0)
        print(f"[trainer] resumed from episode {trainer._episode}")

    trainer.train(num_episodes=999_999)


# ------------------------------------------------------------------
# Slider helpers
# ------------------------------------------------------------------


def _slider_handle_rect(speed: float) -> pygame.Rect:
    frac = (speed - SLIDER_MIN_SPEED) / (SLIDER_MAX_SPEED - SLIDER_MIN_SPEED)
    frac = max(0.0, min(1.0, frac))
    hx = int(SLIDER_X + frac * SLIDER_W)
    return pygame.Rect(hx - 6, SLIDER_Y - 6, 12, 12)


def _draw_controls(
    surface: pygame.Surface,
    paused: bool,
    speed: float,
    font: pygame.font.Font,
    checkpoint_episode: int,
):
    # Slider track
    pygame.draw.line(
        surface,
        (80, 80, 100),
        (SLIDER_X, SLIDER_Y),
        (SLIDER_X + SLIDER_W, SLIDER_Y),
        2,
    )
    pygame.draw.rect(surface, (140, 140, 180), _slider_handle_rect(speed))

    labels = [
        f"{'[PAUSED]' if paused else '[RUNNING]'}",
        f"Speed: {speed:.3f}  (+/- to adjust)",
        f"R: new instance  Ctrl+R: restart  SPACE: pause",
        f"Checkpoint episode: {checkpoint_episode}",
    ]
    for i, text in enumerate(labels):
        surf = font.render(text, True, (120, 120, 140))
        surface.blit(surf, (SLIDER_X + SLIDER_W + 20, SLIDER_Y - 20 + i * 16))


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def run(num_nodes: int = config.NUM_NODES):
    # --- Start trainer thread ---
    trainer_thread = threading.Thread(
        target=_run_trainer,
        args=(CHECKPOINT_PATH, num_nodes),
        daemon=True,
    )
    trainer_thread.start()
    print("[main] trainer thread started")

    # --- Build visualization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If a checkpoint exists already, load it; otherwise start from scratch
    p = Path(CHECKPOINT_PATH)
    if p.exists():
        viz = Visualization.load_agent(
            CHECKPOINT_PATH, num_nodes=num_nodes, speed=DEFAULT_SPEED, device=device
        )
        print(f"[main] loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        # No checkpoint yet — build a fresh untrained agent
        agent = TransformerAgent(
            node_features=4, state_features=3, d_model=128, device=device
        )
        viz = Visualization(
            agent=agent, num_nodes=num_nodes, device=device, speed=DEFAULT_SPEED
        )
        print("[main] no checkpoint found, starting with untrained agent")

    viz.reset()  # uses default seed → reproducible instance

    # Track checkpoint mtime for hot-reload
    last_mtime: float = p.stat().st_mtime if p.exists() else 0.0
    last_poll_time: float = time.time()
    checkpoint_episode: int = 0
    pending_reload: bool = False  # True when a new checkpoint is waiting to load

    # --- Pygame setup ---
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
    pygame.display.set_caption("VRP — Live Training Visualization")
    clock = pygame.time.Clock()
    Sprites.init()

    graph_rect = pygame.Rect(0, 0, config.GRAPH_W, config.WINDOW_H)
    hud_rect = pygame.Rect(config.GRAPH_W, 0, config.HUD_W, config.WINDOW_H)
    renderer = SimulationUI(screen, graph_rect)
    hud = HUD(screen, hud_rect)
    font_small = pygame.font.SysFont("monospace", config.FONT_SIZE_SMALL)

    speed = DEFAULT_SPEED
    paused = False
    step_once = False
    dragging_slider = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    while True:
        clock.tick(config.FPS_CAP)

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    pygame.quit()
                    return

                if event.key == pygame.K_SPACE:
                    paused = not paused

                if event.key == pygame.K_RIGHT and paused:
                    step_once = True

                if event.key == pygame.K_r:
                    if ctrl:
                        viz.reset()  # same instance
                    else:
                        viz.randomize()  # new random instance

                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    speed = min(speed + SPEED_STEP, SPEED_MAX)
                    viz._speed = speed

                if event.key == pygame.K_MINUS:
                    speed = max(speed - SPEED_STEP, SPEED_MIN)
                    viz._speed = speed

            if event.type == pygame.MOUSEBUTTONDOWN:
                if _slider_handle_rect(speed).collidepoint(event.pos):
                    dragging_slider = True

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False

            if event.type == pygame.MOUSEMOTION and dragging_slider:
                rel_x = max(SLIDER_X, min(event.pos[0], SLIDER_X + SLIDER_W))
                frac = (rel_x - SLIDER_X) / SLIDER_W
                speed = SLIDER_MIN_SPEED + frac * (SLIDER_MAX_SPEED - SLIDER_MIN_SPEED)
                viz._speed = speed

        # --- Checkpoint hot-reload ---
        now = time.time()
        if now - last_poll_time >= POLL_INTERVAL_S:
            last_poll_time = now
            if p.exists():
                mtime = p.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    pending_reload = True

        # --- Simulation ---
        if not paused or step_once:
            step_once = False

            if viz.is_done():
                if pending_reload:
                    try:
                        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
                        viz.agent.encoder.load_state_dict(ckpt["encoder"])
                        viz.agent.decoder.load_state_dict(ckpt["decoder"])
                        viz.agent.eval()
                        checkpoint_episode = ckpt.get("episode", 0)
                        print(
                            f"[main] reloaded checkpoint (episode {checkpoint_episode})"
                        )
                    except Exception as e:
                        print(f"[main] checkpoint reload failed: {e}")
                    finally:
                        pending_reload = False
                viz.reset()

            viz.microstep()

        # --- Render ---
        screen.fill((0, 0, 0))
        snapshot = viz.current_snapshot()
        renderer.draw(snapshot)
        hud.draw(snapshot)
        _draw_controls(screen, paused, speed, font_small, checkpoint_episode)
        pygame.display.flip()


if __name__ == "__main__":
    run()
