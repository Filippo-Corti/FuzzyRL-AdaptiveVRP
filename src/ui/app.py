from __future__ import annotations

import json
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pygame
import torch

import config
from src.agent import FuzzyAgent, TransformerAgent
from src.env.batch_env import BatchVRPEnv
from src.training import FuzzyTrainer, TransformerTrainer
from src.visualization import FuzzyVisualization, TransformerVisualization
from src.visualization.base import BaseVisualization
from src.ui.hud import HUD
from src.ui.renderer import VRPRenderer
from src.ui.sprites import Sprites


@dataclass
class AppConfig:
    agent_mode: Literal["transformer", "fuzzy"]
    checkpoint_path: str
    poll_interval_s: float
    default_speed: float
    speed_step: float
    speed_min: float
    speed_max: float
    trainer_batch_size: int = 128
    trainer_save_every: int = 100
    trainer_torch_threads: int = 1
    seed: int | None = None


class SimulationApp:
    """Owns training thread lifecycle, checkpoint reloads, and the PyGame UI loop."""

    def __init__(self, app_config: AppConfig, num_nodes: int = config.NUM_NODES):
        self.cfg = app_config
        self.num_nodes = num_nodes

        self.checkpoint_path = Path(self.cfg.checkpoint_path)
        self.viz: BaseVisualization | None = None
        self.loaded_checkpoint_path: Path | None = None
        self.loaded_checkpoint_episode: int = -1

        self.checkpoint_episode: float = 0.0
        self.pending_reload: bool = False
        self.pending_reload_path: Path | None = None
        self.pending_reload_episode: int = -1
        self.last_poll_time: float = time.time()
        self.training_status_path = _training_status_path(self.checkpoint_path)
        self.training_episode: int = -1
        self.training_baseline: float | None = None
        self.training_adv_ema: float | None = None
        self._training_status_mtime: float = 0.0

        self.speed = self.cfg.default_speed
        self.paused = False
        self.step_once = False
        self._trainer_process: mp.Process | None = None

    def run(self) -> None:
        """Starts the trainer and enters the PyGame UI loop"""
        self._start_trainer()
        self.viz = self._build_visualization()

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
        pygame.display.set_caption("VRP - Live Visualization")
        clock = pygame.time.Clock()
        Sprites.init()

        graph_rect = pygame.Rect(0, 0, config.GRAPH_W, config.WINDOW_H)
        hud_rect = pygame.Rect(config.GRAPH_W, 0, config.HUD_W, config.WINDOW_H)
        renderer = VRPRenderer(screen, graph_rect)
        hud = HUD(screen, hud_rect)

        while True:
            clock.tick(config.FPS_CAP)

            for event in pygame.event.get():
                renderer.handle_event(event)
                hud_action = hud.handle_event(event)

                if event.type == pygame.QUIT:
                    self._stop_trainer()
                    pygame.quit()
                    return

                if hud_action == "toggle_pause":
                    self.paused = not self.paused
                elif hud_action == "step_once" and self.paused:
                    self.step_once = True
                elif hud_action == "speed_up":
                    self.speed = min(self.speed + self.cfg.speed_step, self.cfg.speed_max)
                    self.viz.set_speed(self.speed)
                elif hud_action == "speed_down":
                    self.speed = max(self.speed - self.cfg.speed_step, self.cfg.speed_min)
                    self.viz.set_speed(self.speed)

            self._poll_checkpoint_updates()
            self._poll_training_status()

            if not self.paused:
                if self.viz.is_done():
                    self._handle_reload_if_needed(self.viz)
                    self.viz.reset()
                self.viz.microstep()
            elif self.step_once:
                self.step_once = False
                if self.viz.is_done():
                    self._handle_reload_if_needed(self.viz)
                    self.viz.reset()
                self.viz.step_once()

            screen.fill((0, 0, 0))
            snapshot = self.viz.current_snapshot()
            renderer.draw(snapshot)
            hud.set_training_stats(
                self.training_episode,
                self.training_baseline,
                self.training_adv_ema,
            )
            hud.draw(
                snapshot,
                self.paused,
                self.speed,
                self.checkpoint_episode,
                self.training_episode,
            )
            pygame.display.flip()

    def _start_trainer(self) -> None:
        """Start the training loop in a separate process."""
        # Reset in-memory status and remove stale status file from previous runs.
        self.training_episode = -1
        self.training_baseline = None
        self.training_adv_ema = None
        self._training_status_mtime = 0.0
        try:
            if self.training_status_path.exists():
                self.training_status_path.unlink()
        except OSError:
            pass

        self._trainer_process = mp.Process(
            target=_run_trainer_worker,
            args=(
                self.cfg.agent_mode,
                str(self.checkpoint_path),
                self.num_nodes,
                self.cfg.trainer_batch_size,
                self.cfg.trainer_save_every,
                self.cfg.trainer_torch_threads,
            ),
            daemon=True,
        )
        self._trainer_process.start()
        print(f"[main] trainer process started (pid={self._trainer_process.pid})")

    def _stop_trainer(self) -> None:
        """Gracefully stop the trainer process if it's running."""
        if self._trainer_process is not None and self._trainer_process.is_alive():
            self._trainer_process.terminate()
            self._trainer_process.join(timeout=1.0)
            self._trainer_process = None

    def _build_visualization(self) -> BaseVisualization:
        """Build the visualization instance, loading the latest checkpoint if available."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        viz: BaseVisualization
        latest_checkpoint, latest_episode = _find_latest_checkpoint(
            self.checkpoint_path
        )

        if self.cfg.agent_mode == "transformer":
            if latest_checkpoint is not None:
                viz = TransformerVisualization.load_agent(
                    str(latest_checkpoint),
                    num_nodes=self.num_nodes,
                    speed=self.cfg.default_speed,
                    device=device,
                    seed=self.cfg.seed,
                )
                print(f"[main] loaded checkpoint from {latest_checkpoint}")
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
                    seed=self.cfg.seed,
                )
                print("[main] no checkpoint found, starting with untrained agent")
        else:
            cpu_device = torch.device("cpu")
            if latest_checkpoint is not None:
                viz = FuzzyVisualization.load_agent(
                    str(latest_checkpoint),
                    num_nodes=self.num_nodes,
                    speed=self.cfg.default_speed,
                    device=cpu_device,
                    seed=self.cfg.seed,
                )
                print(f"[main] loaded fuzzy checkpoint from {latest_checkpoint}")
            else:
                fuzzy_agent = FuzzyAgent()
                viz = FuzzyVisualization(
                    agent=fuzzy_agent,
                    num_nodes=self.num_nodes,
                    device=cpu_device,
                    speed=self.cfg.default_speed,
                    seed=self.cfg.seed,
                )
                print("[main] no fuzzy checkpoint found, starting fresh")

        self.loaded_checkpoint_path = latest_checkpoint
        self.loaded_checkpoint_episode = latest_episode
        self.checkpoint_episode = float(max(latest_episode, 0))
        viz.reset()
        return viz

    def _poll_checkpoint_updates(self) -> None:
        now = time.time()
        if now - self.last_poll_time < self.cfg.poll_interval_s:
            return

        self.last_poll_time = now
        latest_checkpoint, latest_episode = _find_latest_checkpoint(
            self.checkpoint_path
        )
        if latest_checkpoint is None:
            return

        mtime = latest_checkpoint.stat().st_mtime
        is_newer_episode = latest_episode > self.loaded_checkpoint_episode
        is_newer_file = (
            self.loaded_checkpoint_path is None
            or latest_checkpoint != self.loaded_checkpoint_path
        )

        if is_newer_episode and is_newer_file:
            self.last_mtime = mtime
            self.pending_reload = True
            self.pending_reload_path = latest_checkpoint
            self.pending_reload_episode = latest_episode


    def _poll_training_status(self) -> None:
        if not self.training_status_path.exists():
            return
        mtime = self.training_status_path.stat().st_mtime
        if mtime <= self._training_status_mtime:
            return
        self._training_status_mtime = mtime

        try:
            data = json.loads(self.training_status_path.read_text(encoding="utf-8"))
        except Exception:
            return

        ep_obj = data.get("episode")
        if isinstance(ep_obj, int):
            self.training_episode = ep_obj

        baseline_obj = data.get("baseline")
        if isinstance(baseline_obj, (int, float)):
            self.training_baseline = float(baseline_obj)
        else:
            self.training_baseline = None

        adv_ema_obj = data.get("adv_ema")
        if isinstance(adv_ema_obj, (int, float)):
            self.training_adv_ema = float(adv_ema_obj)
        else:
            self.training_adv_ema = None

    def _handle_reload_if_needed(self, viz: BaseVisualization) -> None:
        if not self.pending_reload:
            return
        if self.pending_reload_path is None:
            self.pending_reload = False
            return

        try:
            loaded_metric = viz.reload_checkpoint(
                str(self.pending_reload_path),
                device=viz.device,
            )
            if self.pending_reload_episode >= 0:
                self.checkpoint_episode = float(self.pending_reload_episode)
            else:
                self.checkpoint_episode = loaded_metric
            self.loaded_checkpoint_path = self.pending_reload_path
            self.loaded_checkpoint_episode = self.pending_reload_episode
            print(f"[main] reloaded checkpoint (episode {self.checkpoint_episode})")
        except Exception as e:
            print(f"[main] checkpoint reload failed: {e}")
        finally:
            self.pending_reload = False
            self.pending_reload_path = None
            self.pending_reload_episode = -1


def _run_trainer_worker(
    agent_mode: Literal["transformer", "fuzzy"],
    checkpoint_path: str,
    num_nodes: int,
    batch_size: int,
    save_every: int,
    torch_threads: int,
) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    checkpoint = Path(checkpoint_path)
    status_path = _training_status_path(checkpoint)

    def _emit_progress(metrics: dict[str, int | float | None]) -> None:
        _write_training_status(status_path, metrics)

    resume_checkpoint, _ = _find_latest_checkpoint(checkpoint)
    trainer: TransformerTrainer | FuzzyTrainer

    if agent_mode == "transformer":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if resume_checkpoint is not None:
            trainer = TransformerTrainer.load(
                str(resume_checkpoint),
                d_model=128,
                batch_size=batch_size,
                num_nodes=num_nodes,
                save_path=str(checkpoint),
                save_every=save_every,
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
                num_nodes=num_nodes,
                device=device,
            )
            trainer = TransformerTrainer(
                agent=agent,
                env=env,
                save_path=str(checkpoint),
                save_every=save_every,
            )
    else:
        device = torch.device("cpu")
        if resume_checkpoint is not None:
            trainer = FuzzyTrainer.load(
                str(resume_checkpoint),
                num_nodes=num_nodes,
                save_path=str(checkpoint),
                save_every=save_every,
                device=device,
            )
            print(f"[fuzzy trainer] resumed from episode {trainer.episode}")
        else:
            fuzzy_agent = FuzzyAgent()
            env = BatchVRPEnv(
                batch_size=1,
                num_nodes=num_nodes,
                device=device,
            )
            trainer = FuzzyTrainer(
                agent=fuzzy_agent,
                env=env,
                save_path=str(checkpoint),
                save_every=save_every,
            )

    trainer.train(num_episodes=100_000, progress_callback=_emit_progress)


def _extract_episode_from_name(base_checkpoint: Path, candidate: Path) -> int | None:
    """Extracts the episode number from a checkpoint filename if it matches the expected pattern."""
    pattern = re.compile(
        rf"^{re.escape(base_checkpoint.stem)}-(\d+){re.escape(base_checkpoint.suffix)}$"
    )
    match = pattern.match(candidate.name)
    if not match:
        return None
    return int(match.group(1))


def _find_latest_checkpoint(base_checkpoint: Path) -> tuple[Path | None, int]:
    """Searches for the latest checkpoint file matching the pattern and returns its path and episode number."""
    best_path: Path | None = None
    best_episode = -1

    if base_checkpoint.parent.exists():
        glob_pattern = f"{base_checkpoint.stem}-*{base_checkpoint.suffix}"
        for candidate in base_checkpoint.parent.glob(glob_pattern):
            episode = _extract_episode_from_name(base_checkpoint, candidate)
            if episode is None:
                continue
            if episode > best_episode:
                best_episode = episode
                best_path = candidate

    # Legacy fallback (old non-versioned checkpoint file)
    if best_path is None and base_checkpoint.exists():
        return base_checkpoint, 0

    return best_path, best_episode


def _training_status_path(base_checkpoint: Path) -> Path:
    return base_checkpoint.with_name(f"{base_checkpoint.stem}-training-status.json")


def _write_training_status(
    path: Path,
    metrics: dict[str, int | float | None],
) -> None:
    payload = {
        "episode": metrics.get("episode"),
        "baseline": metrics.get("baseline"),
        "adv_ema": metrics.get("adv_ema"),
        "updated_at": time.time(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")

    # On Windows, readers can transiently lock the destination file.
    # Retry atomic replace a few times, then fall back to best-effort write.
    for _ in range(5):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.01)

    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except PermissionError:
        # Ignore transient lock contention; next update will retry.
        pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
