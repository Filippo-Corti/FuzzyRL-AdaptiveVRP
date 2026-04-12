import pygame

from . import colors


class Sprites:

    truck = pygame.image.load("assets/sprites/truck.png") # 219x80 pixels
    house = pygame.image.load("assets/sprites/house.png") # 163x128 pixels

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        cls.truck = pygame.transform.scale(cls.truck.convert_alpha(), (80, 30))
        cls.house = pygame.transform.scale(cls.house.convert_alpha(), (60, 84))

    @classmethod
    def draw_truck(
        cls,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        color: pygame.Color,
        heading_deg: float = 0.0,
        broken: bool = False,
        fraction: float = 0.0,
    ):
        x, y = int(pos.x), int(pos.y)

        truck_surface = cls.tint_image(cls.truck, color)

        # Draw load bar directly on truck surface so it rotates with the truck.
        bar_w, bar_h = 43, 26
        bx, by = 35, 2
        pygame.draw.rect(truck_surface, colors.HUD_BAR_BG, (bx, by, bar_w, bar_h))
        fill_w = int(bar_w * min(max(fraction, 0), 1))
        if fill_w > 0:
            pygame.draw.rect(truck_surface, color, (bx, by, fill_w, bar_h))

        # If broken, draw an X over the bar area; this also rotates with the truck.
        if broken:
            pygame.draw.line(
                truck_surface,
                colors.TRUCK_BROKEN,
                (bx, by),
                (bx + bar_w, by + bar_h),
                3,
            )
            pygame.draw.line(
                truck_surface,
                colors.TRUCK_BROKEN,
                (bx + bar_w, by),
                (bx, by + bar_h),
                3,
            )

        # Sprite default orientation is left-facing, so apply a 180deg offset.
        rotated_sprite = pygame.transform.rotate(truck_surface, heading_deg + 180.0)
        rect = rotated_sprite.get_rect(center=(x, y))
        surface.blit(rotated_sprite, rect)

    @classmethod
    def draw_depot(
        cls,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        color: pygame.Color,
        size_x: int = 70,
        size_y: int = 40,
    ):
        x, y = int(pos[0]), int(pos[1])
        rect = pygame.Rect(x - size_x // 2, y - size_y // 2, size_x, size_y)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=3)
        font = pygame.font.SysFont("monospace", 16)
        label = font.render("DEPOT", True, (255, 255, 255))
        label_rect = label.get_rect(center=rect.center)
        surface.blit(label, label_rect)

    @classmethod
    def draw_node(
        cls,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        color: pygame.Color,
        radius: int = 10,
    ):
        _ = radius
        x, y = int(pos[0]), int(pos[1])
        tinted_sprite = cls.tint_image(cls.house, color)
        rect = tinted_sprite.get_rect(center=(x, y))
        surface.blit(tinted_sprite, rect)

    @staticmethod
    def tint_image(image, color):
        tinted = image.copy()
        tinted.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
        return tinted
