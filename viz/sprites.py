import pygame


def draw_truck(
    surface: pygame.Surface,
    pos: pygame.Vector2,
    color: pygame.Color,
    radius: int = 10,
    broken: bool = False,
):
    x, y = int(pos[0]), int(pos[1])
    pygame.draw.circle(surface, color, (x, y), radius)
    pygame.draw.circle(surface, (255, 255, 255), (x, y), radius, 2)
    if broken:
        d = radius - 2
        pygame.draw.line(surface, (220, 60, 60), (x - d, y - d), (x + d, y + d), 2)
        pygame.draw.line(surface, (220, 60, 60), (x + d, y - d), (x - d, y + d), 2)


def draw_depot(
    surface: pygame.Surface,
    pos: pygame.Vector2,
    color: pygame.Color,
    size: int = 14,
):
    x, y = int(pos[0]), int(pos[1])
    rect = pygame.Rect(x - size // 2, y - size // 2, size, size)
    pygame.draw.rect(surface, color, rect, border_radius=3)
    pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=3)


def draw_node(
    surface: pygame.Surface, pos: pygame.Vector2, color: pygame.Color, radius: int = 6
):
    pygame.draw.circle(surface, color, (int(pos[0]), int(pos[1])), radius)
