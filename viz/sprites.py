import pygame


def draw_truck(
    surface: pygame.Surface,
    pos: pygame.Vector2,
    color: pygame.Color,
    size_x: int = 50,
    size_y: int = 30,
    broken: bool = False,
):
    x, y = int(pos[0]), int(pos[1])
    rect = pygame.Rect(x - size_x // 2, y - size_y // 2, size_x, size_y)
    pygame.draw.rect(surface, color, rect, width=5)

    pygame.draw.line(
        surface,
        color,
        (x - size_x // 2, y - size_y // 2),
        (x + size_x // 2, y + size_y // 2),
        4,
    )
    pygame.draw.line(
        surface,
        color,
        (x + size_x // 2, y - size_y // 2),
        (x - size_x // 2, y + size_y // 2),
        4,
    )


def draw_depot(
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


def draw_node(
    surface: pygame.Surface, pos: pygame.Vector2, color: pygame.Color, radius: int = 10
):
    pygame.draw.circle(surface, color, (int(pos[0]), int(pos[1])), radius)
