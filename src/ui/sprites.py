from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import math

import pygame
import torch


class VisualizationSprites:
    """Draws VRP visual sprites (routes, nodes, depot, truck) onto a pygame surface."""

    _fancy_mode = False
    _house_sprite_base: pygame.Surface | None = None
    _depot_sprite_base: pygame.Surface | None = None
    _truck_sprite_base: pygame.Surface | None = None
    _house_sprite_tinted: dict[tuple[int, int, int], pygame.Surface] = {}
    _demand_font: pygame.font.Font | None = None

    @classmethod
    def set_fancy_mode(cls, enabled: bool) -> None:
        cls._fancy_mode = bool(enabled)

    @classmethod
    def initialize_assets(cls) -> None:
        if (
            cls._house_sprite_base is not None
            and cls._depot_sprite_base is not None
            and cls._truck_sprite_base is not None
        ):
            return

        house_path = Path(__file__).resolve().parents[2] / "assets" / "sprites" / "house.png"
        depot_path = Path(__file__).resolve().parents[2] / "assets" / "sprites" / "depot.png"
        truck_path = Path(__file__).resolve().parents[2] / "assets" / "sprites" / "truck.png"

        loaded = pygame.image.load(str(house_path))
        depot_loaded = pygame.image.load(str(depot_path))
        truck_loaded = pygame.image.load(str(truck_path))
        if pygame.display.get_surface() is not None:
            loaded = loaded.convert_alpha()
            depot_loaded = depot_loaded.convert_alpha()
            truck_loaded = truck_loaded.convert_alpha()
        cls._house_sprite_base = pygame.transform.smoothscale(loaded, (20, 28))
        cls._depot_sprite_base = pygame.transform.smoothscale(depot_loaded, (38, 28))
        cls._truck_sprite_base = pygame.transform.smoothscale(truck_loaded, (34, 14))
        cls._house_sprite_tinted.clear()
        if pygame.font.get_init():
            cls._demand_font = pygame.font.SysFont("segoe ui", 12, bold=True)

    @classmethod
    def _get_tinted_house(cls, color: tuple[int, int, int]) -> pygame.Surface | None:
        if cls._house_sprite_base is None:
            return None
        if color not in cls._house_sprite_tinted:
            tinted = cls._house_sprite_base.copy()
            tinted.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
            cls._house_sprite_tinted[color] = tinted
        return cls._house_sprite_tinted[color]

    @staticmethod
    def draw_routes(
        surface: pygame.Surface,
        routes: list[tuple[float, float]],
        to_screen: Callable[[float, float], tuple[int, int]],
        depot_xy: torch.Tensor,
        route_palette: list[tuple[int, int, int]],
        default_route_color: tuple[int, int, int] = (42, 114, 201),
    ) -> None:
        if len(routes) < 2:
            return

        depot_x = float(depot_xy[0].item())
        depot_y = float(depot_xy[1].item())

        def is_depot(pt: tuple[float, float]) -> bool:
            return math.hypot(pt[0] - depot_x, pt[1] - depot_y) <= 1e-6

        trip_segments: list[list[tuple[float, float]]] = []
        current_trip: list[tuple[float, float]] = [routes[0]]

        for pt in routes[1:]:
            current_trip.append(pt)
            if is_depot(pt):
                if len(current_trip) >= 2:
                    trip_segments.append(current_trip)
                current_trip = [pt]

        if len(current_trip) >= 2:
            trip_segments.append(current_trip)

        for idx, trip in enumerate(trip_segments):
            color = (
                route_palette[idx % len(route_palette)]
                if route_palette
                else default_route_color
            )
            route_pts = [to_screen(float(x), float(y)) for x, y in trip]
            width = 4 if VisualizationSprites._fancy_mode else 3
            pygame.draw.lines(surface, color, False, route_pts, width=width)

    @staticmethod
    def draw_nodes(
        surface: pygame.Surface,
        node_xy: torch.Tensor,
        node_demands: torch.Tensor,
        node_urgency_levels: torch.Tensor,
        visited: torch.Tensor,
        visited_late_mask: torch.Tensor | None,
        appearances: torch.Tensor,
        current_timestep: torch.Tensor,
        to_screen: Callable[[float, float], tuple[int, int]],
    ) -> None:
        for idx in range(node_xy.shape[0]):
            x = float(node_xy[idx, 0].item())
            y = float(node_xy[idx, 1].item())
            sx, sy = to_screen(x, y)

            is_visited = bool(visited[idx].item())
            is_visible = bool((appearances[idx] <= current_timestep).item())
            if is_visited:
                color = (188, 188, 188)
            elif not is_visible:
                color = (216, 216, 216)
            else:
                color = (244, 244, 244)

            if VisualizationSprites._fancy_mode:
                sprite = VisualizationSprites._get_tinted_house(color)
                if sprite is not None:
                    rect = sprite.get_rect(center=(sx, sy))
                    surface.blit(sprite, rect)
                    if VisualizationSprites._demand_font is not None:
                        demand = int(round(float(node_demands[idx].item())))
                        label = VisualizationSprites._demand_font.render(
                            str(demand),
                            True,
                            (38, 38, 38),
                        )
                        label_pos = (sx + 12, sy - 18)
                        surface.blit(label, label_pos)

                        if is_visited:
                            show_late = (
                                visited_late_mask is not None
                                and bool(visited_late_mask[idx].item())
                            )
                            if show_late:
                                late_label = VisualizationSprites._demand_font.render(
                                    "L",
                                    True,
                                    (196, 38, 38),
                                )
                                late_pos = (sx + 9, sy + 10)
                                surface.blit(late_label, late_pos)
                        else:
                            urgency = float(node_urgency_levels[idx].item())
                            urgency_color = (
                                (196, 38, 38) if urgency < 0.0 else (78, 78, 78)
                            )
                            urgency_label = VisualizationSprites._demand_font.render(
                                f"{urgency:.1f}",
                                True,
                                urgency_color,
                            )
                            # Pedix/subscript style placement: under and slightly right of the sprite center.
                            urgency_pos = (sx + 7, sy + 10)
                            surface.blit(urgency_label, urgency_pos)
                    continue
            pygame.draw.circle(surface, color, (sx, sy), 6)

    @staticmethod
    def draw_depot(
        surface: pygame.Surface,
        depot_xy: torch.Tensor,
        to_screen: Callable[[float, float], tuple[int, int]],
    ) -> None:
        depot_sx, depot_sy = to_screen(
            float(depot_xy[0].item()),
            float(depot_xy[1].item()),
        )
        if VisualizationSprites._fancy_mode and VisualizationSprites._depot_sprite_base is not None:
            rect = VisualizationSprites._depot_sprite_base.get_rect(center=(depot_sx, depot_sy))
            surface.blit(VisualizationSprites._depot_sprite_base, rect)
            return

        depot_shape = [
            (depot_sx, depot_sy - 10),
            (depot_sx + 9, depot_sy),
            (depot_sx, depot_sy + 10),
            (depot_sx - 9, depot_sy),
        ]
        pygame.draw.polygon(surface, (235, 235, 235), depot_shape)
        pygame.draw.polygon(surface, (158, 158, 158), depot_shape, width=2)

    @staticmethod
    def draw_truck(
        surface: pygame.Surface,
        truck_xy: torch.Tensor,
        remaining_capacity: torch.Tensor,
        total_capacity: torch.Tensor,
        to_screen: Callable[[float, float], tuple[int, int]],
    ) -> None:
        truck_sx, truck_sy = to_screen(
            float(truck_xy[0].item()),
            float(truck_xy[1].item()),
        )

        if VisualizationSprites._fancy_mode and VisualizationSprites._truck_sprite_base is not None:
            y_offset = 6
            sprite_rect = VisualizationSprites._truck_sprite_base.get_rect(
                center=(truck_sx, truck_sy + y_offset)
            )
            surface.blit(VisualizationSprites._truck_sprite_base, sprite_rect)

            total = max(float(total_capacity.item()), 1e-8)
            remaining = max(0.0, min(float(remaining_capacity.item()), total))
            remaining_ratio = remaining / total
            used_ratio = 1.0 - remaining_ratio

            bar_w = 28
            bar_h = 4
            bar_x = truck_sx - bar_w // 2
            bar_y = sprite_rect.bottom + 3
            bar_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
            pygame.draw.rect(surface, (230, 230, 230), bar_rect, border_radius=2)

            used_w = int(round(bar_w * used_ratio))
            if used_w > 0:
                used_rect = pygame.Rect(bar_x, bar_y, used_w, bar_h)
                pygame.draw.rect(surface, (120, 120, 120), used_rect, border_radius=2)

            pygame.draw.rect(surface, (95, 95, 95), bar_rect, width=1, border_radius=2)
            return

        pygame.draw.circle(surface, (248, 248, 248), (truck_sx, truck_sy), 8)
        pygame.draw.circle(surface, (118, 118, 118), (truck_sx, truck_sy), 8, width=2)
