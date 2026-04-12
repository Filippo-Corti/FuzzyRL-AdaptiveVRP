from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pygame
import torch

import config
from src.agent import FuzzyAgent, TransformerAgent
from src.env.batch_env import BatchVRPEnv
from src.training import FuzzyTrainer, TransformerTrainer
from src.visualization import FuzzyVisualization, TransformerVisualization
from src.visualization.base import BaseVisualization
from src.viz.hud import HUD
from src.viz.renderer import SimulationUI
from src.viz.sprites import Sprites


@dataclass
class AppConfig:
    agent_mode: Literal["transformer", "fuzzy"]
    checkpoint_path: str
    poll_interval_s: float
    default_speed: float
    speed_step: float
    speed_min: float
    speed_max: float


class SimulationApp:
    """Owns training thread lifecycle, checkpoint reloads, and the PyGame UI loop."""

    def __init__(self, app_config: AppConfig, num_nodes: int = config.NUM_NODES):
        self.cfg = app_config
        self.num_nodes = num_nodes

        self.checkpoint_path = Path(self.cfg.checkpoint_path)
        self.viz: BaseVisualization | None = None

        self.checkpoint_episode: float = 0.0
        self.pending_reload: bool = False
        self.last_mtime: float = (
            self.checkpoint_path.stat().st_mtime if self.checkpoint_path.exists() else 0.0
        )
        self.last_poll_time: float = time.time()

        self.speed = self.cfg.default_speed
        self.paused = False
        self.step_once = False
        self.dragging_slider = False

    def _trainer_loop(self, batch_size: int = 128) -> None:
        trainer: TransformerTrainer | FuzzyTrainer
        if self.cfg.agent_mode == "transformer":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.checkpoint_path.exists():
                trainer = TransformerTrainer.load(
                    str(self.checkpoint_path),
                    d_model=128,
                    batch_size=batch_size,
                    num_nodes=self.num_nodes,
                    save_path=str(self.checkpoint_path),
                    save_every=20,
                    device=device,
                )
                print(f"[trainer] resumed from episode {trainer.episode}")
            else:
                agent = TransformerAgent(
                    node_features=4,
                    state_features=3,
                    d_model=128,
                    device=device,
                )
                env = BatchVRPEnv(
                    batch_size=batch_size,
                    num_nodes=self.num_nodes,
                    device=device,
                )
                trainer = TransformerTrainer(
                    agent=agent,
                    env=env,
                    save_path=str(self.checkpoint_path),
                    save_every=20,
                )
        else:
            device = torch.device("cpu")
            if self.checkpoint_path.exists():
                trainer = FuzzyTrainer.load(
                    str(self.checkpoint_path),
                    num_nodes=self.num_nodes,
                    save_path=str(self.checkpoint_path),
                    save_every=20,
                    device=device,
                )
                print(f"[fuzzy trainer] resumed from episode {trainer.episode}")
            else:
                fuzzy_agent = FuzzyAgent()
                env = BatchVRPEnv(batch_size=1, num_nodes=self.num_nodes, device=device)
                trainer = FuzzyTrainer(
                    agent=fuzzy_agent,
                    env=env,
                    save_path=str(self.checkpoint_path),
                    save_every=20,
                )

        trainer.train(num_episodes=100_000)

    def _start_trainer_thread(self) -> None:
        trainer_thread = threading.Thread(target=self._trainer_loop, daemon=True)
        trainer_thread.start()
        print("[main] trainer thread started")

    def _build_visualization(self) -> BaseVisualization:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        viz: BaseVisualization

        if self.cfg.agent_mode == "transformer":
            if self.checkpoint_path.exists():
                viz = TransformerVisualization.load_agent(
                    str(self.checkpoint_path),
                    num_nodes=self.num_nodes,
                    speed=self.cfg.default_speed,
                    device=device,
                )
                print(f"[main] loaded checkpoint from {self.checkpoint_path}")
            else:
                agent = TransformerAgent(
                    node_features=4,
                    state_features=3,
                    d_model=128,
                    device=device,
                )
                viz = TransformerVisualization(
                    agent=agent,
                    num_nodes=self.num_nodes,
                    device=device,
                    speed=self.cfg.default_speed,
                )
                print("[main] no checkpoint found, starting with untrained agent")
        else:
            cpu_device = torch.device("cpu")
            if self.checkpoint_path.exists():
                viz = FuzzyVisualization.load_agent(
                    str(self.checkpoint_path),
                    num_nodes=self.num_nodes,
                    speed=self.cfg.default_speed,
                    device=cpu_device,
                )
                print(f"[main] loaded fuzzy checkpoint from {self.checkpoint_path}")
            else:
                fuzzy_agent = FuzzyAgent()
                viz = FuzzyVisualization(
                    agent=fuzzy_agent,
                    num_nodes=self.num_nodes,
                    device=cpu_device,
                    speed=self.cfg.default_speed,
                )
                print("[main] no fuzzy checkpoint found, starting fresh")

        viz.reset()
        return viz

    def _slider_handle_rect(self) -> pygame.Rect:
        frac = (self.speed - self.cfg.speed_min) / (
            config.SLIDER_MAX_SPEED - self.cfg.speed_min
        )
        frac = max(0.0, min(1.0, frac))
        hx = int(config.SLIDER_X + frac * config.SLIDER_W)
        slider_y = config.WINDOW_H - config.SLIDER_Y_OFFSET
        return pygame.Rect(hx - 6, slider_y - 6, 12, 12)

    def _draw_controls(self, surface: pygame.Surface, font: Any) -> None:
        slider_y = config.WINDOW_H - config.SLIDER_Y_OFFSET

        pygame.draw.line(
            surface,
            (80, 80, 100),
            (config.SLIDER_X, slider_y),
            (config.SLIDER_X + config.SLIDER_W, slider_y),
            2,
        )
        pygame.draw.rect(surface, (140, 140, 180), self._slider_handle_rect())

        labels = [
            f"{'[PAUSED]' if self.paused else '[RUNNING]'}",
            f"Speed: {self.speed:.3f}  (+/- to adjust)",
            "R: new instance  Ctrl+R: restart  SPACE: pause",
            f"Checkpoint episode: {self.checkpoint_episode}",
        ]
        for i, text in enumerate(labels):
            surf = font.render(text, True, (120, 120, 140))
            surface.blit(
                surf,
                (config.SLIDER_X + config.SLIDER_W + 20, slider_y - 20 + i * 16),
            )

    def _poll_checkpoint_updates(self) -> None:
        now = time.time()
        if now - self.last_poll_time < self.cfg.poll_interval_s:
            return

        self.last_poll_time = now
        if not self.checkpoint_path.exists():
            return

        mtime = self.checkpoint_path.stat().st_mtime
        if mtime > self.last_mtime:
            self.last_mtime = mtime
            self.pending_reload = True

    def _handle_reload_if_needed(self, viz: BaseVisualization) -> None:
        if not self.pending_reload:
            return

        try:
            self.checkpoint_episode = viz.reload_checkpoint(
                str(self.checkpoint_path),
                device=viz.device,
            )
            print(f"[main] reloaded checkpoint (episode {self.checkpoint_episode})")
        except Exception as e:
            print(f"[main] checkpoint reload failed: {e}")
        finally:
            self.pending_reload = False

    def run(self) -> None:
        self._start_trainer_thread()
        self.viz = self._build_visualization()

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
        pygame.display.set_caption("VRP - Live Training Visualization")
        clock = pygame.time.Clock()
        Sprites.init()

        graph_rect = pygame.Rect(0, 0, config.GRAPH_W, config.WINDOW_H)
        hud_rect = pygame.Rect(config.GRAPH_W, 0, config.HUD_W, config.WINDOW_H)
        renderer = SimulationUI(screen, graph_rect)
        hud = HUD(screen, hud_rect)
        font_small = pygame.font.SysFont("monospace", config.FONT_SIZE_SMALL)

        while True:
            clock.tick(config.FPS_CAP)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.KEYDOWN:
                    ctrl = pygame.key.get_mods() & pygame.KMOD_CTRL

                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        return
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if event.key == pygame.K_RIGHT and self.paused:
                        self.step_once = True
                    if event.key == pygame.K_r:
                        if ctrl:
                            self.viz.reset()
                        else:
                            self.viz.randomize()
                    if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        self.speed = min(self.speed + self.cfg.speed_step, self.cfg.speed_max)
                        self.viz.set_speed(self.speed)
                    if event.key == pygame.K_MINUS:
                        self.speed = max(self.speed - self.cfg.speed_step, self.cfg.speed_min)
                        self.viz.set_speed(self.speed)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self._slider_handle_rect().collidepoint(event.pos):
                        self.dragging_slider = True

                if event.type == pygame.MOUSEBUTTONUP:
                    self.dragging_slider = False

                if event.type == pygame.MOUSEMOTION and self.dragging_slider:
                    rel_x = max(
                        config.SLIDER_X,
                        min(event.pos[0], config.SLIDER_X + config.SLIDER_W),
                    )
                    frac = (rel_x - config.SLIDER_X) / config.SLIDER_W
                    self.speed = self.cfg.speed_min + frac * (
                        config.SLIDER_MAX_SPEED - self.cfg.speed_min
                    )
                    self.viz.set_speed(self.speed)

            self._poll_checkpoint_updates()

            if not self.paused or self.step_once:
                self.step_once = False

                if self.viz.is_done():
                    self._handle_reload_if_needed(self.viz)
                    self.viz.reset()

                self.viz.microstep()

            screen.fill((0, 0, 0))
            snapshot = self.viz.current_snapshot()
            renderer.draw(snapshot)
            hud.draw(snapshot)
            self._draw_controls(screen, font_small)
            pygame.display.flip()
