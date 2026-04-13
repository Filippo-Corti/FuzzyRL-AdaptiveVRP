import pygame

import config
from . import colors
from .sprites import Sprites
from ..visualization import snapshot


class VRPRenderer:
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
        self.draw_background()
        self.draw_routes(render_state.environment.truck.routes)
        self.draw_nodes(render_state.environment.graph)
        self.draw_depot(render_state.environment.depot)
        self.draw_truck(render_state.environment.truck)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Reserved for renderer-local interactions (panning/hover/select)."""
        _ = event

    def draw_background(self):
        pygame.draw.rect(self.surface, colors.BACKGROUND, self.rect)

    def draw_routes(self, routes: list[list[snapshot.PositionSnapshot]]):
        for i, route in enumerate(routes):
            color = colors.ROUTE_PALETTE[i % len(colors.ROUTE_PALETTE)]
            screen_pts = [self._to_screen(*stop) for stop in route]
            if len(screen_pts) >= 2:
                pygame.draw.lines(self.surface, color, False, screen_pts, 4)

    def draw_nodes(self, nodes: list[snapshot.NodeSnapshot]):
        for node in nodes:
            color = colors.NODE_VISITED if node.visited else colors.NODE_UNVISITED
            sx, sy = self._to_screen(*node.pos)
            Sprites.draw_node(self.surface, pygame.Vector2(sx, sy), color)
            label = self._font_small.render(
                f"({node.pos[0]:.2f}, {node.pos[1]:.2f}) - {node.demand}", True, (140, 140, 160)
            )

            self.surface.blit(label, (sx - label.get_rect().centerx, sy + 8))

    def draw_depot(self, depot: snapshot.DepotSnapshot):
        Sprites.draw_depot(self.surface, self._to_screen(*depot.pos), colors.NODE_DEPOT)

    def draw_truck(self, truck: snapshot.TruckSnapshot):
        color = colors.ROUTE_PALETTE[0]
        Sprites.draw_truck(
            self.surface,
            self._to_screen(*truck.pos),
            color,
            truck.heading_deg,
            broken=False,
            fraction=truck.load / truck.capacity if truck.capacity > 0 else 0,
        )

    def _to_screen(self, nx: float, ny: float, margin: int = 100) -> pygame.Vector2:
        """Map normalised node coordinates [0,1]×[0,1] to screen pixels, leaving a margin."""
        sx = margin + self.rect.left + int(nx * (self.rect.width - 2 * margin))
        sy = margin + self.rect.top + int(ny * (self.rect.height - 2 * margin))
        return pygame.Vector2(sx, sy)
