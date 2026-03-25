import pygame

import config
from simulation import snapshot
from viz import colors


class HUD:
    """
    Renders the right-side information panel.
    Receives agent_info dict rather than the agent directly.
    """

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

    def draw(self, render_state: snapshot.SimulationSnapshot):
        """
        render_state contains a snapshot of the state of the simulation
        """
        self._draw_panel_bg()

        env_stats = render_state.stats
        agent_info = render_state.agent

        y = self.rect.top + config.FONT_SIZE
        y = self._draw_section_header("Episode stats", y)
        y = self._draw_stats(env_stats, y)
        y += config.FONT_SIZE

        y = self._draw_section_header(f"Truck {"N/A"} — Fuzzy State", y)
        y = self._draw_memberships(agent_info.memberships, y)
        y += config.FONT_SIZE

        y = self._draw_section_header("Q-values", y)
        y = self._draw_q_values(agent_info.q_values, agent_info.chosen_action, y)
        y += config.FONT_SIZE

    def _draw_panel_bg(self):
        panel_surf = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        panel_surf.fill(colors.HUD_BG)
        self.surface.blit(panel_surf, self.rect.topleft)

    def _draw_section_header(self, text: str, y: float):
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

    def _draw_stats(self, stats: snapshot.SimulationStats, y: float):
        lines = [
            ("Round", str(stats.round)),
            ("Status", stats.status),
            ("Orphans", f"{stats.orphans} / {stats.total_nodes}"),
            (
                "Active trucks",
                f"{stats.active_trucks} / {stats.total_trucks}",
            ),
            ("Distance", f"{stats.total_distance:.1f}"),
            ("Ep. reward", f"{stats.episode_reward:.2f}"),
            ("Truck turn", f"{stats.truck_turn}"),
            ("Last action", stats.last_action),
            (
                "Best solution",
                f"{stats.best_solution_distance:.1f}",
            ),
            (
                "Last solution",
                f"{stats.last_distance:.1f}",
            ),
        ]
        for label, value in lines:
            self._draw_kv(label, value, self.rect.left, y)
            y += 18
        return y

    def _draw_memberships(self, memberships: dict[str, dict[str, float]], y: float):
        """Draw a mini bar chart for each fuzzy variable."""
        for var_name, labels in memberships.items():
            lbl = self._font_small.render(var_name, True, colors.HUD_TEXT)
            self.surface.blit(lbl, (self.rect.left + 10, y))
            y += config.FONT_SIZE
            for label_name, membership in labels.items():
                y = self._draw_membership_bar(label_name, membership, y)
            y += self.BAR_H * 0.3
        return y

    def _draw_membership_bar(self, label_name: str, value: float, y: float):
        bx = self.rect.left + 20
        # background
        pygame.draw.rect(
            self.surface,
            colors.HUD_BAR_BG,
            (bx, y, self.BAR_W, self.BAR_H),
            border_radius=2,
        )
        # fill
        fill_w = int(self.BAR_W * value)
        bar_color = colors.HUD_BAR_HIGH if value > 0.7 else colors.HUD_BAR_FILL
        if fill_w > 0:
            pygame.draw.rect(
                self.surface, bar_color, (bx, y, fill_w, self.BAR_H), border_radius=2
            )
        # label
        self._draw_kv(label_name, f"{value:.2f}", bx + self.BAR_W, y - 1)
        return y + self.BAR_H + 3

    def _draw_q_values(self, q_values: dict[str, float], chosen_action: str, y: float):
        """Horizontal bar per action, chosen action highlighted."""
        y += 10
        min_q = min(q_values.values(), default=0.0)
        max_q = max(q_values.values(), default=0.0)
        span = max_q - min_q if max_q != min_q else 1.0
        for action, q in q_values.items():
            norm = (q - min_q) / span
            is_chosen = action == chosen_action
            bar_color = (240, 200, 60) if is_chosen else colors.HUD_BAR_FILL
            bx = self.rect.left + 10
            pygame.draw.rect(
                self.surface,
                colors.HUD_BAR_BG,
                (bx, y, self.BAR_W, self.BAR_H),
                border_radius=2,
            )
            fill_w = int(self.BAR_W * norm)
            if fill_w > 0:
                pygame.draw.rect(
                    self.surface,
                    bar_color,
                    (bx, y, fill_w, self.BAR_H),
                    border_radius=2,
                )
            self._draw_kv(
                f"{'+' if is_chosen else ' '} {action}",
                f"{q:.3f}",
                bx + self.BAR_W,
                y - 1,
            )
            y += config.FONT_SIZE
        return y

    def _draw_kv(self, key: str, value: str, x: float, y: float):
        k_surf = self._font_small.render(key + ":", True, (140, 140, 160))
        v_surf = self._font_small.render(value, True, colors.HUD_TEXT)
        self.surface.blit(k_surf, (x + 10, y))
        self.surface.blit(v_surf, (x + 300, y))
