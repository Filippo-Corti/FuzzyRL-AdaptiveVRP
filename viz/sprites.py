import pygame


class Sprites:

    truck = pygame.image.load("assets/sprites/truck.png")

    def __init__(self):
        pass

    @classmethod
    def init(cls):
        cls.truck = pygame.transform.scale(cls.truck.convert_alpha(), (80, 30))

    @classmethod
    def draw_truck(
        cls,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        color: pygame.Color,
        broken: bool = False,
    ):
        x, y = int(pos.x), int(pos.y)

        tinted_sprite = cls.tint_image(cls.truck, color)
        rect = tinted_sprite.get_rect(center=(x, y))
        surface.blit(tinted_sprite, rect)

    @classmethod
    def draw_depot(
        cls,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        color: pygame.Color,
        size_x: int = 60,
        size_y: int = 40,
    ):
        x, y = int(pos[0]), int(pos[1])
        rect = pygame.Rect(x - size_x // 2, y - size_y // 2, size_x, size_y)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=3)

    @classmethod
    def draw_node(
        cls,
        surface: pygame.Surface,
        pos: pygame.Vector2,
        color: pygame.Color,
        radius: int = 10,
    ):
        pygame.draw.circle(surface, color, (int(pos[0]), int(pos[1])), radius)

    @staticmethod
    def tint_image(image, color):
        tinted = image.copy()
        tinted.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
        return tinted
