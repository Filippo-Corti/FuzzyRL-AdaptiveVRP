from __future__ import annotations

import pygame

from .base import BaseHUD


class FuzzyHUD(BaseHUD):
    """HUD for fuzzy training metrics."""

    def __init__(self, surface: pygame.Surface, panel_rect: pygame.Rect):
        super().__init__(surface, panel_rect)
        self._epsilon_points: list[tuple[int, float]] = []

    def _on_new_training_stats(self, episode: int, stats: dict[str, object]) -> None:
        _ = stats
        epsilon = self._get_stat_float("epsilon")
        if epsilon is not None:
            self._epsilon_points.append((episode, epsilon))

    def _action_distribution(self) -> dict[str, float]:
        raw = self._training_stats.get("action_distribution")
        if not isinstance(raw, dict):
            return {}

        parsed: dict[str, float] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                parsed[key] = float(value)
        return parsed

    def draw_training_stats_section(self, section_rect: pygame.Rect) -> None:
        y = self.draw_section_header("Training stats", section_rect.top)
        training_episode = self._get_stat_int("episode")
        training_label = "-" if training_episode is None or training_episode < 0 else str(training_episode)
        self._draw_kv("Episode", training_label, section_rect.left, y)
        y += 22

        epsilon_rect = pygame.Rect(
            section_rect.left + 6,
            y,
            section_rect.width - 12,
            150,
        )
        self.draw_line_plot(
            rect=epsilon_rect,
            points=list(self._epsilon_points),
            color=(116, 188, 255),
            label="Epsilon",
            latest=self._get_stat_float("epsilon"),
            x_axis_label="episodes",
            y_axis_label="epsilon",
        )

        rel_gap_rect = pygame.Rect(
            section_rect.left + 6,
            epsilon_rect.bottom + 8,
            section_rect.width - 12,
            150,
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

        text_y = rel_gap_rect.bottom + 10
        q_table_size = self._get_stat_int("q_table_size")
        q_table_label = "-" if q_table_size is None else str(q_table_size)
        self._draw_kv("Q-table", q_table_label, section_rect.left, text_y)
        text_y += 20

        dist = self._action_distribution()
        if dist:
            preferred_order = ["nearest", "second", "third", "depot"]
            items = [
                (name, dist[name])
                for name in preferred_order
                if name in dist
            ]
            for name, value in dist.items():
                if name not in preferred_order:
                    items.append((name, value))

            self._draw_kv("Actions", "", section_rect.left, text_y)
            text_y += 16
            for name, value in items[:6]:
                pct = 100.0 * value
                self._draw_kv(f"  {name}", f"{pct:.1f}%", section_rect.left, text_y)
                text_y += 16
