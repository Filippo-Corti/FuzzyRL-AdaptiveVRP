from __future__ import annotations

import pygame

from .base import BaseHUD


class TransformerHUD(BaseHUD):
    """HUD for transformer training metrics."""

    def __init__(self, surface: pygame.Surface, panel_rect: pygame.Rect):
        super().__init__(surface, panel_rect)
        self._baseline_points: list[tuple[int, float]] = []

    def _on_new_training_stats(self, episode: int, stats: dict[str, object]) -> None:
        _ = stats
        baseline = self._get_stat_float("baseline")
        if baseline is not None:
            self._baseline_points.append((episode, baseline))

    def draw_training_stats_section(self, section_rect: pygame.Rect) -> None:
        y = self.draw_section_header("Training stats", section_rect.top)
        training_episode = self._get_stat_int("episode")
        training_label = "-" if training_episode is None or training_episode < 0 else str(training_episode)
        self._draw_kv("Episode", training_label, section_rect.left, y)
        y += 22

        baseline_rect = pygame.Rect(
            section_rect.left + 6,
            y,
            section_rect.width - 12,
            200,
        )
        self.draw_line_plot(
            rect=baseline_rect,
            points=list(self._baseline_points),
            color=(102, 214, 177),
            label="Baseline",
            latest=self._get_stat_float("baseline"),
            x_axis_label="episodes",
            y_axis_label="baseline",
        )

        rel_gap_rect = pygame.Rect(
            section_rect.left + 6,
            baseline_rect.bottom + 10,
            section_rect.width - 12,
            200,
        )
        self.draw_line_plot(
            rect=rel_gap_rect,
            points=list(self._gap_ema_points),
            color=(245, 186, 74),
            label="Relative gap",
            latest=self._gap_ema,
            x_axis_label="viz episodes",
            y_axis_label="rel gap",
        )
