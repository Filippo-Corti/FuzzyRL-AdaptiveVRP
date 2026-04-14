from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import pygame

import config
from ...visualization import snapshot
from .. import colors


HUDAction = Literal["toggle_pause", "step_once", "speed_up", "speed_down"]


class BaseHUD(ABC):
    """Abstract HUD with shared controls, instance stats and plotting utilities."""

    PANEL_W = 260
    BAR_H = config.FONT_SIZE_SMALL
    BAR_W = 140

    def __init__(self, surface: pygame.Surface, panel_rect: pygame.Rect):
        self.surface = surface
        self.rect = panel_rect
        self._font: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.FONT_SIZE, bold=True
        )
        self._font_small: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.FONT_SIZE_SMALL
        )
        self._buttons: dict[HUDAction, pygame.Rect] = self._build_buttons()

        self._training_stats: dict[str, object] = {}
        self._last_stats_episode: int = -1

        self._gap_ema: float | None = None
        self._gap_ema_alpha: float = 0.1
        self._gap_ema_points: list[tuple[int, float]] = []
        self._viz_episode_index: int = 0
        self._prev_round: int | None = None
        self._prev_total_distance: float | None = None
        self._prev_exact_cost: float | None = None

    def set_training_stats(self, stats: dict[str, object]) -> None:
        self._training_stats = stats
        episode = self._get_stat_int("episode")
        if episode is not None and episode > self._last_stats_episode:
            self._on_new_training_stats(episode, stats)
            self._last_stats_episode = episode

    def _on_new_training_stats(self, episode: int, stats: dict[str, object]) -> None:
        _ = episode
        _ = stats

    def _get_stat_int(self, key: str) -> int | None:
        value = self._training_stats.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    def _get_stat_float(self, key: str) -> float | None:
        value = self._training_stats.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _build_buttons(self) -> dict[HUDAction, pygame.Rect]:
        x = self.rect.left + 12
        y = self.rect.bottom - 170
        w = self.rect.width - 24
        h = 34
        gap = 8
        return {
            "toggle_pause": pygame.Rect(x, y, w, h),
            "step_once": pygame.Rect(x, y + (h + gap), w, h),
            "speed_up": pygame.Rect(x, y + 2 * (h + gap), (w - gap) // 2, h),
            "speed_down": pygame.Rect(
                x + (w - gap) // 2 + gap,
                y + 2 * (h + gap),
                (w - gap) // 2,
                h,
            ),
        }

    def handle_event(self, event: pygame.event.Event) -> HUDAction | None:
        if event.type != pygame.MOUSEBUTTONDOWN:
            return None
        if event.button != 1:
            return None
        for action, rect in self._buttons.items():
            if rect.collidepoint(event.pos):
                return action
        return None

    def draw(
        self,
        simulation_snapshot: snapshot.SimulationSnapshot,
        paused: bool,
        speed: float,
        checkpoint_episode: float,
    ) -> None:
        self._update_gap_ema_series(simulation_snapshot)
        self.draw_background()

        top_rect, mid_rect, bottom_rect = self._section_rects()
        self.draw_instance_stats_section(simulation_snapshot, top_rect)
        self.draw_training_stats_section(mid_rect)
        self.draw_controls_section(paused, speed, checkpoint_episode, bottom_rect)

    def _update_gap_ema_series(
        self,
        simulation_snapshot: snapshot.SimulationSnapshot,
    ) -> None:
        """Track per-instance relative gap against baseline cost and append EMA on each reset."""
        current_round = simulation_snapshot.stats.round

        if (
            self._prev_round is not None
            and current_round < self._prev_round
            and self._prev_total_distance is not None
            and self._prev_exact_cost is not None
        ):
            denom = max(abs(self._prev_exact_cost), 1e-9)
            gap = (self._prev_total_distance - self._prev_exact_cost) / denom
            if self._gap_ema is None:
                self._gap_ema = gap
            else:
                self._gap_ema = (
                    (1.0 - self._gap_ema_alpha) * self._gap_ema
                    + self._gap_ema_alpha * gap
                )

            self._viz_episode_index += 1
            self._gap_ema_points.append((self._viz_episode_index, self._gap_ema))

        self._prev_round = current_round
        self._prev_total_distance = simulation_snapshot.stats.total_distance
        self._prev_exact_cost = simulation_snapshot.stats.exact_cost

    def _section_rects(self) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        pad = 10
        controls_h = 270
        available_h = self.rect.height - controls_h - 4 * pad
        top_h = max(120, int(available_h * 0.25))
        mid_h = max(120, available_h - top_h)

        top_rect = pygame.Rect(
            self.rect.left + pad,
            self.rect.top + pad,
            self.rect.width - 2 * pad,
            top_h,
        )
        mid_rect = pygame.Rect(
            self.rect.left + pad,
            top_rect.bottom + pad,
            self.rect.width - 2 * pad,
            mid_h,
        )
        bottom_rect = pygame.Rect(
            self.rect.left + pad,
            self.rect.bottom - controls_h - pad,
            self.rect.width - 2 * pad,
            controls_h,
        )
        return top_rect, mid_rect, bottom_rect

    def draw_instance_stats_section(
        self,
        simulation_snapshot: snapshot.SimulationSnapshot,
        section_rect: pygame.Rect,
    ) -> None:
        y = self.draw_section_header("Instance stats", section_rect.top)
        self.draw_stats(simulation_snapshot, y)

    @abstractmethod
    def draw_training_stats_section(self, section_rect: pygame.Rect) -> None:
        """Render mode-specific training stats."""

    def draw_background(self) -> None:
        panel_surf = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        panel_surf.fill(colors.HUD_BG)
        self.surface.blit(panel_surf, self.rect.topleft)

    def draw_section_header(self, text: str, y: float) -> float:
        label = self._font.render(text, True, (180, 180, 200))
        self.surface.blit(label, (self.rect.left + 10, y))
        pygame.draw.line(
            self.surface,
            (60, 60, 80),
            (self.rect.left + 8, y + config.FONT_SIZE),
            (self.rect.right - 8, y + config.FONT_SIZE),
            1,
        )
        return y + config.FONT_SIZE + 5

    def draw_stats(self, simulation_snapshot: snapshot.SimulationSnapshot, y: float) -> float:
        stats = simulation_snapshot.stats
        exact_label = "-" if stats.exact_cost is None else f"{stats.exact_cost:.2f}"
        lines = [
            ("Round", str(stats.round)),
            ("Orphans", f"{stats.orphans} / {stats.total_nodes}"),
            ("Distance", f"{stats.total_distance:.1f}"),
            ("NN cost", exact_label),
            (
                "Truck load",
                f"{simulation_snapshot.environment.truck.load:.1f} / {simulation_snapshot.environment.truck.capacity:.1f}",
            ),
        ]

        for label, value in lines:
            self._draw_kv(label, value, self.rect.left, y)
            y += 18
        return y

    def _draw_kv(self, key: str, value: str, x: float, y: float) -> None:
        k_surf = self._font_small.render(key + ":", True, (140, 140, 160))
        v_surf = self._font_small.render(value, True, colors.HUD_TEXT)
        self.surface.blit(k_surf, (x + 10, y))
        self.surface.blit(v_surf, (x + 170, y))

    def draw_line_plot(
        self,
        rect: pygame.Rect,
        points: list[tuple[int, float]],
        color: tuple[int, int, int],
        label: str,
        latest: float | None,
        x_axis_label: str,
        y_axis_label: str,
    ) -> None:
        pygame.draw.rect(self.surface, (34, 38, 50), rect, border_radius=6)
        pygame.draw.rect(self.surface, (66, 72, 90), rect, width=1, border_radius=6)

        label_value = "-" if latest is None else f"{latest:.3f}"
        title = self._font_small.render(f"{label}: {label_value}", True, colors.HUD_TEXT)
        self.surface.blit(title, (rect.left + 6, rect.top + 4))

        values = [v for _, v in points]
        if values:
            v_min = min(values)
            v_max = max(values)
        else:
            v_min = 0.0
            v_max = 1.0

        if v_max - v_min < 1e-9:
            v_min -= 1.0
            v_max += 1.0

        x_tick_count = 5
        y_tick_fractions = [0.0, 0.2, 0.4, 0.7, 1.0]

        if points:
            x_start = points[0][0]
            x_end = points[-1][0]
        else:
            x_start = 0
            x_end = 1
        x_span = max(1, x_end - x_start)

        x_label_surf = self._font_small.render(x_axis_label, True, (140, 140, 160))
        y_label_surf = self._font_small.render(y_axis_label, True, (140, 140, 160))
        y_label_vertical = pygame.transform.rotate(y_label_surf, 90)

        y_tick_surfaces: list[tuple[float, pygame.Surface]] = []
        y_tick_w = 0
        for f in y_tick_fractions:
            y_tick_value = v_min + f * (v_max - v_min)
            tick_surface = self._font_small.render(
                f"{y_tick_value:.2f}", True, (126, 132, 148)
            )
            y_tick_surfaces.append((f, tick_surface))
            y_tick_w = max(y_tick_w, tick_surface.get_width())

        plot_pad_x = 8
        y_label_gap = 4
        y_tick_gap = 6
        x_tick_len = 4
        x_tick_gap = 2
        x_label_gap = 2
        plot_top = rect.top + 28
        plot_bottom = rect.bottom - (
            6
            + x_tick_len
            + x_tick_gap
            + self._font_small.get_height()
            + x_label_gap
            + x_label_surf.get_height()
        )
        plot_left = (
            rect.left
            + plot_pad_x
            + y_label_vertical.get_width()
            + y_label_gap
            + y_tick_w
            + y_tick_gap
        )
        plot_right = rect.right - plot_pad_x
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        if plot_w <= 2 or plot_h <= 2:
            return

        self.surface.blit(
            x_label_surf,
            (
                plot_left + max(0, (plot_w - x_label_surf.get_width()) // 2),
                plot_bottom
                + x_tick_len
                + x_tick_gap
                + self._font_small.get_height()
                + x_label_gap,
            ),
        )
        self.surface.blit(
            y_label_vertical,
            (
                rect.left + plot_pad_x,
                plot_top + max(0, (plot_h - y_label_vertical.get_height()) // 2),
            ),
        )

        axis_color = (92, 98, 118)
        pygame.draw.line(
            self.surface,
            axis_color,
            (plot_left, plot_top),
            (plot_left, plot_bottom),
            1,
        )
        pygame.draw.line(
            self.surface,
            axis_color,
            (plot_left, plot_bottom),
            (plot_right, plot_bottom),
            1,
        )

        for i in range(x_tick_count):
            f = i / (x_tick_count - 1)
            x = plot_left + f * plot_w
            pygame.draw.line(
                self.surface,
                axis_color,
                (x, plot_bottom),
                (x, plot_bottom + 4),
                1,
            )
            tick_value = int(round(x_start + f * x_span))
            tick_label = self._font_small.render(str(tick_value), True, (126, 132, 148))
            self.surface.blit(
                tick_label,
                (
                    x - tick_label.get_width() // 2,
                    plot_bottom + x_tick_len + x_tick_gap,
                ),
            )

        for f, tick_label in y_tick_surfaces:
            y = plot_bottom - f * plot_h
            pygame.draw.line(
                self.surface,
                axis_color,
                (plot_left - 4, y),
                (plot_left, y),
                1,
            )
            self.surface.blit(
                tick_label,
                (plot_left - 6 - tick_label.get_width(), y - tick_label.get_height() // 2),
            )

        if len(points) < 2:
            return

        line_points: list[tuple[float, float]] = []
        for ep, v in points:
            x = plot_left + ((ep - x_start) / x_span) * plot_w
            t = (v - v_min) / (v_max - v_min)
            y = plot_bottom - t * plot_h
            line_points.append((x, y))

        pygame.draw.lines(self.surface, color, False, line_points, 2)

    def draw_controls_section(
        self,
        paused: bool,
        speed: float,
        checkpoint_episode: float,
        section_rect: pygame.Rect,
    ) -> None:
        _ = self.draw_section_header("Controls", section_rect.top)

        status = "PAUSED" if paused else "RUNNING"
        status_surf = self._font_small.render(
            f"Status: {status}",
            True,
            colors.HUD_TEXT,
        )
        speed_surf = self._font_small.render(
            f"Speed: {speed:.3f}",
            True,
            colors.HUD_TEXT,
        )
        ckpt_surf = self._font_small.render(
            f"Checkpoint: {int(checkpoint_episode)}",
            True,
            colors.HUD_TEXT,
        )

        info_x = section_rect.left + 2
        info_y = section_rect.top + config.FONT_SIZE + 16
        self.surface.blit(status_surf, (info_x, info_y))
        self.surface.blit(speed_surf, (info_x, info_y + 18))
        self.surface.blit(ckpt_surf, (info_x, info_y + 36))

        labels: dict[HUDAction, str] = {
            "toggle_pause": "Resume" if paused else "Pause",
            "step_once": "Step Once",
            "speed_up": "+ Speed",
            "speed_down": "- Speed",
        }

        for action, rect in self._buttons.items():
            pygame.draw.rect(self.surface, (225, 228, 236), rect, border_radius=8)
            pygame.draw.rect(
                self.surface,
                (160, 166, 180),
                rect,
                width=1,
                border_radius=8,
            )
            label = self._font_small.render(labels[action], True, (55, 60, 72))
            self.surface.blit(label, label.get_rect(center=rect.center))
