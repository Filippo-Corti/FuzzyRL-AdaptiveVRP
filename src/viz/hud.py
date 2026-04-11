import pygame

from .. import config
from ..simulation import snapshot
from . import colors


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

    def draw(self, snapshot: snapshot.SimulationSnapshot):
        """
        snapshot contains a snapshot of the state of the simulation
        """
        self.draw_background()

        y = self.rect.top + config.FONT_SIZE
        y = self.draw_section_header("Episode stats", y)
        y = self.draw_stats(snapshot, y)
        y += config.FONT_SIZE

    def draw_background(self):
        panel_surf = pygame.Surface(
            (self.rect.width, self.rect.height), pygame.SRCALPHA
        )
        panel_surf.fill(colors.HUD_BG)
        self.surface.blit(panel_surf, self.rect.topleft)

    def draw_section_header(self, text: str, y: float):
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

    def draw_stats(self, snapshot: snapshot.SimulationSnapshot, y: float):
        stats = snapshot.stats
        lines = [
            ("Round", str(stats.round)),
            ("Orphans", f"{stats.orphans} / {stats.total_nodes}"),
            ("Distance", f"{stats.total_distance:.1f}"),
            (
                "Truck load",
                f"{snapshot.environment.truck.load:.1f} / {snapshot.environment.truck.capacity:.1f}",
            ),
        ]

        for label, value in lines:
            self._draw_kv(label, value, self.rect.left, y)
            y += 18
        return y

    def _draw_kv(self, key: str, value: str, x: float, y: float):
        k_surf = self._font_small.render(key + ":", True, (140, 140, 160))
        v_surf = self._font_small.render(value, True, colors.HUD_TEXT)
        self.surface.blit(k_surf, (x + 10, y))
        self.surface.blit(v_surf, (x + 300, y))
