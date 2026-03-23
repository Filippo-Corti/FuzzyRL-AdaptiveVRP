import pygame

import config
from . import colors
from .sprites import Sprites
from simulation import snapshot


class Renderer:
    """
    Draws the graph, nodes, routes, and truck positions onto a surface.
    """

    def __init__(self, surface: pygame.Surface, graph_rect: pygame.Rect):
        """
        surface    — pygame Surface to draw onto (the main window surface)
        graph_rect — pygame.Rect defining the area allocated to the graph
        """
        self.surface = surface
        self.rect = graph_rect
        self._font_small: pygame.font.Font = pygame.font.SysFont(
            "monospace", config.FONT_SIZE_SMALL
        )

    def draw(self, render_state: snapshot.SimulationSnapshot):
        """
        render_state contains a snapshot of the state of the simulation
        """
        self._draw_background()
        self._draw_routes(render_state.environment.routes)
        self._draw_nodes(render_state.environment.graph)
        self._draw_depot(render_state.environment.depot)
        self._draw_trucks(render_state.environment.trucks)

    def _draw_background(self):
        pygame.draw.rect(self.surface, colors.BACKGROUND, self.rect)

    def _draw_routes(self, routes: list[list[snapshot.PositionSnapshot]]):
        for i, route in enumerate(routes):
            color = colors.ROUTE_PALETTE[i % len(colors.ROUTE_PALETTE)]
            screen_pts = [self._to_screen(*stop) for stop in route]
            pygame.draw.lines(self.surface, color, False, screen_pts, 4)

    def _draw_nodes(self, nodes: list[snapshot.NodeSnapshot]):
        for node in nodes:
            color = {
                snapshot.NodeStatusSnapshot.UNVISITED: colors.NODE_UNVISITED,
                snapshot.NodeStatusSnapshot.ASSIGNED: colors.NODE_ASSIGNED,
                snapshot.NodeStatusSnapshot.VISITED: colors.NODE_VISITED,
            }.get(node.status, colors.NODE_UNVISITED)
            sx, sy = self._to_screen(*node.pos)
            Sprites.draw_node(self.surface, pygame.Vector2(sx, sy), color)
            label = self._font_small.render(
                f"({node.pos[0]}, {node.pos[1]})", True, (140, 140, 160)
            )

            self.surface.blit(label, (sx - label.get_rect().centerx, sy + 8))

    def _draw_depot(self, depot: snapshot.DepotSnapshot):
        if depot is None:
            return
        Sprites.draw_depot(self.surface, self._to_screen(*depot.pos), colors.NODE_DEPOT)

    def _draw_trucks(self, trucks: list[snapshot.TruckSnapshot]):
        for i, truck in enumerate(trucks):
            color = colors.ROUTE_PALETTE[i % len(colors.ROUTE_PALETTE)]
            broken = truck.status == snapshot.TruckStatusSnapshot.BROKEN
            if broken:
                color = colors.TRUCK_BROKEN
            Sprites.draw_truck(
                self.surface,
                self._to_screen(*truck.pos),
                color,
                broken=broken,
                fraction=truck.rel_load,
            )

    def _to_screen(self, nx: float, ny: float, margin: int = 100) -> pygame.Vector2:
        """Map normalised node coordinates [0,1]×[0,1] to screen pixels, leaving a margin."""
        sx = margin + self.rect.left + int(nx * (self.rect.width - 2 * margin))
        sy = margin + self.rect.top + int(ny * (self.rect.height - 2 * margin))
        return pygame.Vector2(sx, sy)
