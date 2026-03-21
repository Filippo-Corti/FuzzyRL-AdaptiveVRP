import pygame

from env import *
from viz import colors, sprites


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

    def draw(self, render_state: SimulationState):
        """
        render_state contains a snapshot of the state of the simulation
        """
        self._draw_background()
        self._draw_nodes(render_state.nodes)
        self._draw_edges(render_state.edges)
        self._draw_routes(render_state.routes)
        self._draw_depot(render_state.depot)
        self._draw_trucks(render_state.trucks)

    def _draw_background(self):
        pygame.draw.rect(self.surface, colors.BACKGROUND, self.rect)

    def _draw_edges(self, edges: list[EdgeState]):
        for edge in edges:
            x1, y1 = edge.pos1
            x2, y2 = edge.pos2
            pygame.draw.line(
                self.surface,
                colors.GRID,
                self._to_screen(x1, y1),
                self._to_screen(x2, y2),
                1,
            )

    def _draw_routes(self, routes: list[list[PositionState]]):
        for i, route in enumerate(routes):
            color = colors.ROUTE_PALETTE[i % len(colors.ROUTE_PALETTE)]
            screen_pts = [self._to_screen(*stop) for stop in route]
            pygame.draw.lines(self.surface, color, False, screen_pts, 2)

    def _draw_nodes(self, nodes: list[NodeState]):
        for node in nodes:
            color = {
                NodeStatusState.UNVISITED: colors.NODE_UNVISITED,
                NodeStatusState.ASSIGNED: colors.NODE_ASSIGNED,
                NodeStatusState.VISITED: colors.NODE_VISITED,
            }.get(node.status, colors.NODE_UNVISITED)
            sprites.draw_node(self.surface, self._to_screen(*node.pos), color)

    def _draw_depot(self, depot: DepotState):
        if depot is None:
            return
        sprites.draw_depot(self.surface, self._to_screen(*depot.pos), colors.NODE_DEPOT)

    def _draw_trucks(self, trucks: list[TruckState]):
        for truck in trucks:
            color = {
                TruckStatusState.ACTIVE: colors.TRUCK_ACTIVE,
                TruckStatusState.BROKEN: colors.TRUCK_BROKEN,
                TruckStatusState.RECOVERING: colors.TRUCK_RECOVERING,
            }.get(truck.status, colors.TRUCK_ACTIVE)
            broken = truck.status == TruckStatusState.BROKEN
            sprites.draw_truck(
                self.surface, self._to_screen(*truck.pos), color, broken=broken
            )
            self._draw_load_bar(*truck.pos, truck.rel_load, color)

    def _draw_load_bar(self, px, py, fraction, color):
        sx, sy = self._to_screen(px, py)
        bar_w, bar_h = 20, 4
        bx, by = sx - bar_w // 2, sy + 13
        pygame.draw.rect(self.surface, colors.HUD_BAR_BG, (bx, by, bar_w, bar_h))
        fill_w = int(bar_w * min(max(fraction, 0), 1))
        if fill_w > 0:
            pygame.draw.rect(self.surface, color, (bx, by, fill_w, bar_h))

    def _to_screen(self, nx: float, ny: float) -> pygame.Vector2:
        """Map normalised node coordinates [0,1]×[0,1] to screen pixels."""
        sx = self.rect.left + int(nx * self.rect.width)
        sy = self.rect.top + int(ny * self.rect.height)
        return pygame.Vector2(sx, sy)
