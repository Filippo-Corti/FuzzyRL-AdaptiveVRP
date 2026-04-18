from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import pygame
import torch

import config
from src.agents import TONNAgent, TransformerAgent
from src.agents.fuzzy import FuzzyAgent
from src.ui.sprites import VisualizationSprites
from src.vrp import VRPEnvironmentBatch, VRPInstance, VRPInstanceBatch


VIS_CHECKPOINTS_DIR = getattr(config, "VIS_CHECKPOINTS_DIR", "checkpoints")
VIS_LATENESS_ALPHA = float(getattr(config, "VIS_LATENESS_ALPHA", 0.2))
VIS_WINDOW_WIDTH = int(getattr(config, "VIS_WINDOW_WIDTH", 1600))
VIS_WINDOW_HEIGHT = int(getattr(config, "VIS_WINDOW_HEIGHT", 900))
VIS_MIN_WINDOW_WIDTH = int(getattr(config, "VIS_MIN_WINDOW_WIDTH", 1100))
VIS_MIN_WINDOW_HEIGHT = int(getattr(config, "VIS_MIN_WINDOW_HEIGHT", 700))
VIS_FPS_CAP = int(getattr(config, "VIS_FPS_CAP", 60))
VIS_DEFAULT_SPEED_STEPS_PER_SECOND = float(
    getattr(config, "VIS_DEFAULT_SPEED_STEPS_PER_SECOND", 3.0)
)
VIS_MIN_SPEED_STEPS_PER_SECOND = float(
    getattr(config, "VIS_MIN_SPEED_STEPS_PER_SECOND", 0.5)
)
VIS_MAX_SPEED_STEPS_PER_SECOND = float(
    getattr(config, "VIS_MAX_SPEED_STEPS_PER_SECOND", 20.0)
)
VIS_SPEED_STEP = float(getattr(config, "VIS_SPEED_STEP", 0.5))
VIS_FONT_FAMILY = str(getattr(config, "VIS_FONT_FAMILY", "consolas"))
VIS_FONT_SMALL_SIZE = int(getattr(config, "VIS_FONT_SMALL_SIZE", 18))
VIS_FONT_SIZE = int(getattr(config, "VIS_FONT_SIZE", 22))
VIS_FONT_LARGE_SIZE = int(getattr(config, "VIS_FONT_LARGE_SIZE", 34))
VIS_FONT_SCALE_MIN = float(getattr(config, "VIS_FONT_SCALE_MIN", 0.85))
VIS_FONT_SCALE_MAX = float(getattr(config, "VIS_FONT_SCALE_MAX", 1.35))
VIS_FANCY_MODE = bool(getattr(config, "VIS_FANCY_MODE", False))
VIS_ROUTE_PALETTE = [
    tuple(color)
    for color in getattr(
        config,
        "VIS_ROUTE_PALETTE",
        [
            (15, 93, 255),
            (255, 77, 77),
            (27, 164, 95),
            (255, 150, 46),
            (140, 94, 255),
            (255, 45, 133),
            (0, 171, 194),
            (199, 113, 19),
            (78, 97, 114),
            (196, 39, 110),
        ],
    )
]

BG_COLOR = (246, 246, 246)
HEADER_BG = (255, 255, 255)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (222, 222, 222)
GRAPH_BG = (252, 252, 252)
GRAPH_BORDER = (228, 228, 228)
TEXT_PRIMARY = (34, 34, 34)
TEXT_SECONDARY = (90, 90, 90)


@dataclass(frozen=True)
class CheckpointChoice:
    label: str
    path: Path | None
    episode: int


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    key: str

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        selected: bool = False,
        enabled: bool = True,
    ) -> None:
        if not enabled:
            bg = (243, 243, 243)
            border = (210, 210, 210)
            fg = (176, 176, 176)
        elif selected:
            bg = (20, 110, 255)
            border = (5, 66, 166)
            fg = (255, 255, 255)
        else:
            bg = (255, 255, 255)
            border = (198, 198, 198)
            fg = (52, 52, 52)

        pygame.draw.rect(surface, bg, self.rect, border_radius=8)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=8)
        text = font.render(self.label, True, fg)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class PygameVisualizationApp:
    """Pygame visualization loop for TONN vs Fuzzy vs Transformer on the same instance."""

    def __init__(
        self,
        checkpoints_dir: str | Path = VIS_CHECKPOINTS_DIR,
        lateness_penalty_alpha: float = VIS_LATENESS_ALPHA,
        num_nodes: int = config.NUM_NODES,
    ) -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.alpha = float(lateness_penalty_alpha)
        self.num_nodes = int(num_nodes)

        self.device = torch.device("cpu")

        self.screen_state: Literal["config", "simulation"] = "config"
        self.running = True

        self.auto_run = False
        self.speed_steps_per_second = VIS_DEFAULT_SPEED_STEPS_PER_SECOND
        self.min_speed = VIS_MIN_SPEED_STEPS_PER_SECOND
        self.max_speed = VIS_MAX_SPEED_STEPS_PER_SECOND
        self.speed_step = VIS_SPEED_STEP
        self._step_accumulator = 0.0

        self.transformer_choices = self._discover_checkpoint_choices(
            stem_prefix="transformer",
            suffix=".pt",
            untrained_label="No training",
        )
        self.fuzzy_choices = self._discover_checkpoint_choices(
            stem_prefix="fuzzy",
            suffix=".pkl",
            untrained_label="No training",
        )
        self.selected_transformer_idx = max(0, len(self.transformer_choices) - 1)
        self.selected_fuzzy_idx = max(0, len(self.fuzzy_choices) - 1)

        self.tonn_agent = TONNAgent(w_d=1.0, w_u=-1.0)
        self.transformer_agent: TransformerAgent | None = None
        self.fuzzy_agent: FuzzyAgent | None = None

        self.env_tonn: VRPEnvironmentBatch | None = None
        self.env_fuzzy: VRPEnvironmentBatch | None = None
        self.env_transformer: VRPEnvironmentBatch | None = None
        self._late_visited_tonn: torch.Tensor | None = None
        self._late_visited_fuzzy: torch.Tensor | None = None
        self._late_visited_transformer: torch.Tensor | None = None

        self._font_small: pygame.font.Font | None = None
        self._font: pygame.font.Font | None = None
        self._font_large: pygame.font.Font | None = None

        self._buttons: list[Button] = []
        self.fancy_mode = VIS_FANCY_MODE
        VisualizationSprites.set_fancy_mode(self.fancy_mode)

        self._window_w = VIS_WINDOW_WIDTH
        self._window_h = VIS_WINDOW_HEIGHT

    def run(self) -> None:
        pygame.init()
        pygame.font.init()

        screen = pygame.display.set_mode(
            (self._window_w, self._window_h),
            pygame.RESIZABLE,
        )
        VisualizationSprites.initialize_assets()
        pygame.display.set_caption("OnlineVRP - Agent Comparison Visualizer")
        self._rebuild_fonts(self._window_w, self._window_h)

        clock = pygame.time.Clock()

        while self.running:
            dt = clock.tick(VIS_FPS_CAP) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    screen = self._handle_resize(event.w, event.h)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if self.screen_state == "simulation":
                self._update_simulation(dt)

            self._draw(screen)
            pygame.display.flip()

        pygame.quit()

    def _handle_resize(self, width: int, height: int) -> pygame.Surface:
        self._window_w = max(int(width), VIS_MIN_WINDOW_WIDTH)
        self._window_h = max(int(height), VIS_MIN_WINDOW_HEIGHT)
        self._rebuild_fonts(self._window_w, self._window_h)
        return pygame.display.set_mode(
            (self._window_w, self._window_h), pygame.RESIZABLE
        )

    def _rebuild_fonts(self, width: int, height: int) -> None:
        scale = min(width / float(VIS_WINDOW_WIDTH), height / float(VIS_WINDOW_HEIGHT))
        scale = max(VIS_FONT_SCALE_MIN, min(VIS_FONT_SCALE_MAX, scale))

        small_size = max(12, int(round(VIS_FONT_SMALL_SIZE * scale)))
        normal_size = max(14, int(round(VIS_FONT_SIZE * scale)))
        large_size = max(22, int(round(VIS_FONT_LARGE_SIZE * scale)))

        self._font_small = pygame.font.SysFont(VIS_FONT_FAMILY, small_size)
        self._font = pygame.font.SysFont(VIS_FONT_FAMILY, normal_size)
        self._font_large = pygame.font.SysFont(VIS_FONT_FAMILY, large_size, bold=True)

    def _discover_checkpoint_choices(
        self,
        stem_prefix: str,
        suffix: str,
        untrained_label: str,
    ) -> list[CheckpointChoice]:
        all_paths = [
            p
            for p in self.checkpoints_dir.glob(f"{stem_prefix}-*{suffix}")
            if p.is_file() and self._episode_from_stem(p.stem) is not None
        ]
        all_paths.sort(key=lambda p: self._episode_from_stem(p.stem) or -1)

        base_choice = CheckpointChoice(label=untrained_label, path=None, episode=0)
        if not all_paths:
            return [base_choice]

        best = all_paths[-1]
        middle_pool = all_paths[:-1]
        middle = self._pick_equally_spaced(middle_pool, target_count=3)

        selected: list[Path] = []
        for item in middle + [best]:
            if item not in selected:
                selected.append(item)

        selected = selected[:4]

        choices = [base_choice]
        for p in selected:
            episode = self._episode_from_stem(p.stem)
            if episode is None:
                continue
            choices.append(CheckpointChoice(label=p.stem, path=p, episode=episode))
        return choices

    def _pick_equally_spaced(self, paths: list[Path], target_count: int) -> list[Path]:
        if target_count <= 0 or not paths:
            return []
        if len(paths) <= target_count:
            return list(paths)

        episodes = [self._episode_from_stem(p.stem) for p in paths]
        if any(ep is None for ep in episodes):
            return paths[:target_count]

        ep_values = [int(ep) for ep in episodes if ep is not None]
        ep_min = min(ep_values)
        ep_max = max(ep_values)
        if ep_max <= ep_min:
            return paths[:target_count]

        picked: list[Path] = []
        for i in range(1, target_count + 1):
            q = i / (target_count + 1)
            target_ep = ep_min + q * (ep_max - ep_min)

            best_path: Path | None = None
            best_dist: float | None = None
            for p in paths:
                if p in picked:
                    continue
                ep = self._episode_from_stem(p.stem)
                if ep is None:
                    continue
                dist = abs(float(ep) - target_ep)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_path = p

            if best_path is not None:
                picked.append(best_path)

        if len(picked) < target_count:
            for p in paths:
                if p not in picked:
                    picked.append(p)
                if len(picked) >= target_count:
                    break
        return picked[:target_count]

    def _episode_from_stem(self, stem: str) -> int | None:
        match = re.search(r"(\d+)$", stem)
        if not match:
            return None
        return int(match.group(1))

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for button in self._buttons:
            if not button.hit(pos):
                continue

            if self.screen_state == "config":
                if button.key.startswith("cfg:transformer:"):
                    self.selected_transformer_idx = int(button.key.split(":")[-1])
                elif button.key.startswith("cfg:fuzzy:"):
                    self.selected_fuzzy_idx = int(button.key.split(":")[-1])
                elif button.key == "cfg:start":
                    self._start_simulation_from_config()
            else:
                if button.key == "sim:back":
                    self.screen_state = "config"
                    self.auto_run = False
                elif button.key == "sim:fancy":
                    self.fancy_mode = not self.fancy_mode
                    VisualizationSprites.set_fancy_mode(self.fancy_mode)
                elif button.key == "sim:step":
                    if self._has_step_remaining():
                        self._step_once_all()
                elif button.key == "sim:auto":
                    if self._has_step_remaining():
                        self.auto_run = not self.auto_run
                elif button.key == "sim:speed_plus":
                    self.speed_steps_per_second = min(
                        self.max_speed,
                        self.speed_steps_per_second + self.speed_step,
                    )
                elif button.key == "sim:speed_minus":
                    self.speed_steps_per_second = max(
                        self.min_speed,
                        self.speed_steps_per_second - self.speed_step,
                    )
                elif button.key == "sim:new":
                    self._reset_with_new_instance()
            return

    def _start_simulation_from_config(self) -> None:
        self.transformer_agent = self._load_transformer_agent(
            self.transformer_choices[self.selected_transformer_idx].path
        )
        self.fuzzy_agent = self._load_fuzzy_agent(
            self.fuzzy_choices[self.selected_fuzzy_idx].path
        )

        self._reset_with_new_instance()
        self.screen_state = "simulation"
        self.auto_run = False
        self._step_accumulator = 0.0

    def _load_transformer_agent(self, checkpoint_path: Path | None) -> TransformerAgent:
        if checkpoint_path is None:
            agent = TransformerAgent(
                node_features=config.TRANSFORMER_NODE_FEATURES,
                state_features=config.TRANSFORMER_STATE_FEATURES,
                d_model=config.TRANSFORMER_D_MODEL,
                device=self.device,
            )
            agent.eval()
            return agent

        agent = TransformerAgent.load(checkpoint_path, device=self.device)
        agent.eval()
        return agent

    def _load_fuzzy_agent(self, checkpoint_path: Path | None) -> FuzzyAgent:
        if checkpoint_path is None:
            agent = FuzzyAgent(device=self.device)
            agent.eval()
            return agent

        agent = FuzzyAgent.load(checkpoint_path, device=self.device)
        agent.eval()
        return agent

    def _build_single_instance(self) -> VRPInstance:
        return VRPInstance(
            num_nodes=self.num_nodes,
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

    def _clone_instance(self, base: VRPInstanceBatch) -> VRPInstance:
        return VRPInstance.from_batch(base, 0)

    def _reset_with_new_instance(self) -> None:
        base = self._build_single_instance()
        self.env_tonn = VRPEnvironmentBatch(
            instance_batch=self._clone_instance(base),
            lateness_penalty_alpha=self.alpha,
        )
        self.env_fuzzy = VRPEnvironmentBatch(
            instance_batch=self._clone_instance(base),
            lateness_penalty_alpha=self.alpha,
        )
        self.env_transformer = VRPEnvironmentBatch(
            instance_batch=self._clone_instance(base),
            lateness_penalty_alpha=self.alpha,
        )
        self._late_visited_tonn = torch.zeros(
            self.num_nodes,
            dtype=torch.bool,
            device=self.device,
        )
        self._late_visited_fuzzy = torch.zeros(
            self.num_nodes,
            dtype=torch.bool,
            device=self.device,
        )
        self._late_visited_transformer = torch.zeros(
            self.num_nodes,
            dtype=torch.bool,
            device=self.device,
        )

    def _execute_and_track_late(
        self,
        env: VRPEnvironmentBatch,
        actions: torch.Tensor,
        late_mask: torch.Tensor,
    ) -> None:
        prev_visited = env.visited.clone()
        service_t = env.timestep.clone()
        env.execute(actions)

        newly_visited = env.visited & (~prev_visited)
        deadlines = env.instance.appearances + env.instance.window_lengths
        late_now = newly_visited & (service_t.unsqueeze(1) > deadlines)
        late_mask |= late_now[0]

    def _step_once_all(self) -> None:
        if (
            self.env_tonn is None
            or self.env_fuzzy is None
            or self.env_transformer is None
            or self.fuzzy_agent is None
            or self.transformer_agent is None
        ):
            return

        if not self._has_step_remaining():
            return

        if not bool(self.env_tonn.done.all().item()):
            actions_tonn = self.tonn_agent.select_actions(self.env_tonn)
            if self._late_visited_tonn is not None:
                self._execute_and_track_late(
                    env=self.env_tonn,
                    actions=actions_tonn,
                    late_mask=self._late_visited_tonn,
                )
            else:
                self.env_tonn.execute(actions_tonn)

        if not bool(self.env_fuzzy.done.all().item()):
            actions_fuzzy = self.fuzzy_agent.select_actions(self.env_fuzzy, greedy=True)
            if self._late_visited_fuzzy is not None:
                self._execute_and_track_late(
                    env=self.env_fuzzy,
                    actions=actions_fuzzy,
                    late_mask=self._late_visited_fuzzy,
                )
            else:
                self.env_fuzzy.execute(actions_fuzzy)

        if not bool(self.env_transformer.done.all().item()):
            actions_transformer = self.transformer_agent.select_actions(
                self.env_transformer,
                greedy=True,
            )
            if self._late_visited_transformer is not None:
                self._execute_and_track_late(
                    env=self.env_transformer,
                    actions=actions_transformer,
                    late_mask=self._late_visited_transformer,
                )
            else:
                self.env_transformer.execute(actions_transformer)

    def _all_done(self) -> bool:
        if (
            self.env_tonn is None
            or self.env_fuzzy is None
            or self.env_transformer is None
        ):
            return False
        return (
            bool(self.env_tonn.done.all().item())
            and bool(self.env_fuzzy.done.all().item())
            and bool(self.env_transformer.done.all().item())
        )

    def _update_simulation(self, dt: float) -> None:
        if not self.auto_run:
            return
        if not self._has_step_remaining():
            self.auto_run = False
            self._step_accumulator = 0.0
            return
        self._step_accumulator += dt * self.speed_steps_per_second
        while self._step_accumulator >= 1.0:
            self._step_accumulator -= 1.0
            self._step_once_all()

    def _has_step_remaining(self) -> bool:
        return not self._all_done()

    def _draw(self, surface: pygame.Surface) -> None:
        if self._font is None or self._font_small is None or self._font_large is None:
            return

        surface.fill(BG_COLOR)
        self._buttons = []

        if self.screen_state == "config":
            self._draw_config_screen(surface)
        else:
            self._draw_simulation_screen(surface)

    def _draw_config_screen(self, surface: pygame.Surface) -> None:
        assert self._font_large is not None
        assert self._font is not None
        assert self._font_small is not None

        title = self._font_large.render("VRP Visualization Setup", True, TEXT_PRIMARY)
        surface.blit(title, (40, 32))

        subtitle = self._font_small.render(
            "Pick checkpoints for Fuzzy and Transformer, then start.",
            True,
            TEXT_SECONDARY,
        )
        surface.blit(subtitle, (44, 76))

        left = 48
        width = surface.get_width() - 96
        h = surface.get_height()
        row_h = max(56, int(h * 0.082))
        first_row_top = max(140, int(h * 0.2))
        section_gap = max(90, int(h * 0.17))
        second_row_top = first_row_top + section_gap
        start_y = min(h - 190, second_row_top + row_h + 56)

        self._draw_checkpoint_row(
            surface=surface,
            title="Transformer choices (1 to 5):",
            choices=self.transformer_choices,
            selected_idx=self.selected_transformer_idx,
            key_prefix="cfg:transformer",
            top=first_row_top,
            left=left,
            width=width,
            row_height=row_h,
        )
        self._draw_checkpoint_row(
            surface=surface,
            title="Fuzzy choices (1 to 5):",
            choices=self.fuzzy_choices,
            selected_idx=self.selected_fuzzy_idx,
            key_prefix="cfg:fuzzy",
            top=second_row_top,
            left=left,
            width=width,
            row_height=row_h,
        )

        start_rect = pygame.Rect(left, start_y, 260, max(54, int(h * 0.065)))
        start_button = Button(start_rect, "Start Visualization", "cfg:start")
        start_button.draw(surface, self._font, selected=True)
        self._buttons.append(start_button)

        info_lines = [
            f"Selected Transformer: {self.transformer_choices[self.selected_transformer_idx].label}",
            f"Selected Fuzzy: {self.fuzzy_choices[self.selected_fuzzy_idx].label}",
            "Simulation screen controls: step, auto-run, speed +/-, new instance.",
        ]
        y = start_rect.bottom + 24
        for line in info_lines:
            if y > h - 26:
                break
            txt = self._font_small.render(line, True, TEXT_SECONDARY)
            surface.blit(txt, (left, y))
            y += 28

    def _draw_checkpoint_row(
        self,
        surface: pygame.Surface,
        title: str,
        choices: list[CheckpointChoice],
        selected_idx: int,
        key_prefix: str,
        top: int,
        left: int,
        width: int,
        row_height: int,
    ) -> None:
        assert self._font is not None
        assert self._font_small is not None

        lbl = self._font.render(title, True, TEXT_PRIMARY)
        surface.blit(lbl, (left, top - 44))

        gap = 10
        button_w = max(140, (width - gap * (len(choices) - 1)) // max(1, len(choices)))
        x = left
        for idx, choice in enumerate(choices):
            rect = pygame.Rect(x, top, button_w, row_height)
            text = f"{idx + 1}. {choice.label}"
            button = Button(rect, text, f"{key_prefix}:{idx}")
            button.draw(surface, self._font_small, selected=(idx == selected_idx))
            self._buttons.append(button)
            x += button_w + gap

    def _draw_simulation_screen(self, surface: pygame.Surface) -> None:
        assert self._font is not None
        assert self._font_small is not None
        assert self._font_large is not None

        header_h = 56
        controls_h = 112
        body_top = header_h
        body_bottom = surface.get_height() - controls_h
        body_h = body_bottom - body_top

        pygame.draw.rect(
            surface, HEADER_BG, pygame.Rect(0, 0, surface.get_width(), header_h)
        )
        pygame.draw.rect(
            surface,
            PANEL_BORDER,
            pygame.Rect(0, 0, surface.get_width(), header_h),
            width=1,
        )
        title = self._font.render("TONN vs Fuzzy vs Transformer", True, TEXT_PRIMARY)
        surface.blit(title, (20, 16))

        status = "Auto" if self.auto_run else "Manual"
        speed_text = self._font_small.render(
            f"Mode: {status} | Speed: {self.speed_steps_per_second:.1f} steps/s",
            True,
            TEXT_SECONDARY,
        )
        surface.blit(speed_text, (580, 20))

        thirds = surface.get_width() // 3
        panels = [
            (
                "TONN",
                self.env_tonn,
                (58, 58, 58),
                pygame.Rect(0, body_top, thirds, body_h),
            ),
            (
                "Fuzzy",
                self.env_fuzzy,
                (58, 58, 58),
                pygame.Rect(thirds, body_top, thirds, body_h),
            ),
            (
                "Transformer",
                self.env_transformer,
                (58, 58, 58),
                pygame.Rect(
                    2 * thirds, body_top, surface.get_width() - 2 * thirds, body_h
                ),
            ),
        ]

        for name, env, color, rect in panels:
            late_mask = None
            if name == "TONN":
                late_mask = self._late_visited_tonn
            elif name == "Fuzzy":
                late_mask = self._late_visited_fuzzy
            elif name == "Transformer":
                late_mask = self._late_visited_transformer
            self._draw_panel(surface, name, env, rect, color, late_mask)

        self._draw_sim_controls(surface, controls_h)

    def _draw_sim_controls(self, surface: pygame.Surface, controls_h: int) -> None:
        assert self._font is not None

        y = surface.get_height() - controls_h + 14
        x = 18
        h = 56
        gap = 12
        widths = [130, 120, 100, 120, 60, 60, 160]
        specs = [
            ("Back", "sim:back"),
            ("Fancy", "sim:fancy"),
            ("Step", "sim:step"),
            ("Auto", "sim:auto"),
            ("-", "sim:speed_minus"),
            ("+", "sim:speed_plus"),
            ("New Instance", "sim:new"),
        ]

        can_step = self._has_step_remaining()

        for w, (label, key) in zip(widths, specs):
            rect = pygame.Rect(x, y, w, h)
            button = Button(rect, label, key)
            selected = (key == "sim:auto" and self.auto_run) or (
                key == "sim:fancy" and self.fancy_mode
            )
            enabled = True
            if key in {"sim:step", "sim:auto"}:
                enabled = can_step
            button.draw(surface, self._font, selected=selected, enabled=enabled)
            self._buttons.append(button)
            x += w + gap

    def _draw_panel(
        self,
        surface: pygame.Surface,
        name: str,
        env: VRPEnvironmentBatch | None,
        rect: pygame.Rect,
        accent: tuple[int, int, int],
        late_visited_mask: torch.Tensor | None,
    ) -> None:
        assert self._font is not None
        assert self._font_small is not None

        pygame.draw.rect(surface, PANEL_BG, rect)
        pygame.draw.rect(surface, PANEL_BORDER, rect, width=1)

        title = self._font.render(name, True, accent)
        surface.blit(title, (rect.left + 10, rect.top + 8))

        if env is None:
            return

        pad_top = 42
        line_h = max(16, self._font_small.get_height() + 2)
        stats_area_h = line_h * 5 + 16
        draw_rect = pygame.Rect(
            rect.left + 8,
            rect.top + pad_top,
            rect.width - 16,
            max(120, rect.height - pad_top - stats_area_h - 10),
        )

        self._draw_env_graph(surface, env, draw_rect, accent, late_visited_mask)

        total_distance = float(env.total_distance[0].item())
        total_lateness = float(env.total_lateness[0].item())
        steps = int(env.timestep[0].item())
        done = bool(env.done[0].item())

        lines = [
            f"distance: {total_distance:.2f}",
            f"lateness: {total_lateness:.2f}",
            f"combined: {total_distance + self.alpha * total_lateness:.2f}",
            f"step: {steps}",
            "done: yes" if done else "done: no",
        ]

        text_y = draw_rect.bottom + 10
        for line in lines:
            txt = self._font_small.render(line, True, TEXT_SECONDARY)
            surface.blit(txt, (rect.left + 10, text_y))
            text_y += line_h

    def _draw_env_graph(
        self,
        surface: pygame.Surface,
        env: VRPEnvironmentBatch,
        draw_rect: pygame.Rect,
        route_color: tuple[int, int, int],
        late_visited_mask: torch.Tensor | None,
    ) -> None:
        instance = env.instance
        node_xy = instance.node_xy[0]
        node_demands = instance.node_weights[0]
        depot_xy = instance.depot_xy[0]
        visited = env.visited[0]
        appearances = instance.appearances[0]
        window_lengths = instance.window_lengths[0].to(torch.float32)
        current_timestep = env.timestep[0]

        # Signed urgency indicator: positive means time remains, negative means overdue.
        current_t = current_timestep.to(torch.float32)
        deadlines = appearances.to(torch.float32) + window_lengths
        node_urgency_levels = (deadlines - current_t) / (window_lengths + 1e-8)

        low, high = instance.node_xy_range
        span = max(high - low, 1e-8)

        def to_screen(x: float, y: float) -> tuple[int, int]:
            nx = (x - low) / span
            ny = (y - low) / span
            sx = int(draw_rect.left + 14 + nx * (draw_rect.width - 28))
            sy = int(draw_rect.top + 14 + ny * (draw_rect.height - 28))
            return sx, sy

        pygame.draw.rect(surface, GRAPH_BG, draw_rect, border_radius=6)
        pygame.draw.rect(surface, GRAPH_BORDER, draw_rect, width=1, border_radius=6)

        VisualizationSprites.draw_routes(
            surface=surface,
            routes=env.routes[0],
            to_screen=to_screen,
            depot_xy=depot_xy,
            route_palette=VIS_ROUTE_PALETTE,
            default_route_color=route_color,
        )
        VisualizationSprites.draw_nodes(
            surface=surface,
            node_xy=node_xy,
            node_demands=node_demands,
            node_urgency_levels=node_urgency_levels,
            visited=visited,
            visited_late_mask=late_visited_mask,
            appearances=appearances,
            current_timestep=current_timestep,
            to_screen=to_screen,
        )
        VisualizationSprites.draw_depot(
            surface=surface,
            depot_xy=depot_xy,
            to_screen=to_screen,
        )
        VisualizationSprites.draw_truck(
            surface=surface,
            truck_xy=env.truck_xy[0],
            remaining_capacity=env.remaining_cap[0],
            total_capacity=instance.W[0],
            to_screen=to_screen,
        )
