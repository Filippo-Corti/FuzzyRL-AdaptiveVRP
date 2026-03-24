import pygame

from simulation import VRPSimulation
from .plots import MetricsPlotter
from .sprites import Sprites
from .renderer import Renderer
from .hud import HUD

import config

SLIDER_X = 10
SLIDER_Y = 40
SLIDER_W = 180
SLIDER_H = 8
SLIDER_MIN_MS = 0
SLIDER_MAX_MS = 500


def run(simulation: VRPSimulation):
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
    pygame.display.set_caption("VRP Fuzzy Q-Learning")
    clock = pygame.time.Clock()
    Sprites.init()

    graph_rect = pygame.Rect(0, 0, config.GRAPH_W, config.WINDOW_H)
    hud_rect = pygame.Rect(config.GRAPH_W, 0, config.HUD_W, config.WINDOW_H)

    renderer = Renderer(screen, graph_rect)
    hud = HUD(screen, hud_rect)
    plotter = MetricsPlotter()

    paused = False
    step_once = False
    show_best = False
    ms_since_tick = 0
    sim_step_ms = config.SIM_STEP_MS
    dragging_slider = False

    tick = 0
    while True:
        dt = clock.tick(config.FPS_CAP)

        # ── Event handling ──────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT and paused:
                    step_once = True
                if event.key == pygame.K_b:
                    show_best = not show_best
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

            if event.type == pygame.MOUSEBUTTONDOWN:
                if _slider_handle_rect(sim_step_ms).collidepoint(event.pos):
                    dragging_slider = True

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_slider = False

            if event.type == pygame.MOUSEMOTION and dragging_slider:
                rel_x = max(SLIDER_X, min(event.pos[0], SLIDER_X + SLIDER_W))
                frac = (rel_x - SLIDER_X) / SLIDER_W
                sim_step_ms = int(
                    SLIDER_MIN_MS + frac * (SLIDER_MAX_MS - SLIDER_MIN_MS)
                )

        # ── Simulation tick ─────────────────────────────────────────
        if not paused or step_once:
            ms_since_tick += dt
            if ms_since_tick >= sim_step_ms or step_once:
                ms_since_tick = 0
                step_once = False
                simulation.step()

        # ── Render ──────────────────────────────────────────────────
        screen.fill((0, 0, 0))

        snapshot = (
            simulation.best_snapshot
            if show_best and simulation.best_snapshot is not None
            else simulation.get_snapshot()
        )

        renderer.draw(snapshot)
        hud.draw(snapshot)

        if not show_best:
            plotter.update(snapshot)

        if tick % config.PLOT_EVERY == 0:
            plotter.draw()

        _draw_controls(screen, paused, show_best, sim_step_ms)
        pygame.display.flip()

        tick += 1


def _slider_handle_rect(sim_step_ms: int) -> pygame.Rect:
    frac = (sim_step_ms - SLIDER_MIN_MS) / (SLIDER_MAX_MS - SLIDER_MIN_MS)
    handle_x = int(SLIDER_X + frac * SLIDER_W)
    return pygame.Rect(handle_x - 6, SLIDER_Y - 6, 12, SLIDER_H + 12)


def _draw_controls(screen, paused: bool, show_best: bool, sim_step_ms: int):
    font = pygame.font.SysFont("monospace", 11)

    # Hints
    hints = [
        "SPACE  pause / resume",
        "->     step (when paused)",
        "B      toggle best solution",
        "ESC    quit",
    ]
    for i, h in enumerate(hints):
        color = (180, 180, 100) if (h.startswith("B") and show_best) else (80, 80, 100)
        surf = font.render(h, True, color)
        screen.blit(surf, (10, config.WINDOW_H - 18 * (len(hints) - i)))

    # Paused indicator
    if paused:
        lbl = font.render("PAUSED", True, (240, 180, 60))
        screen.blit(lbl, (10, 10))

    # Best solution indicator
    if show_best:
        lbl = font.render("BEST SOLUTION", True, (60, 240, 120))
        screen.blit(lbl, (10, 10) if not paused else (80, 10))

    # Speed slider
    speed_label = font.render("speed", True, (120, 120, 140))
    screen.blit(speed_label, (SLIDER_X, SLIDER_Y - 18))

    # Track
    pygame.draw.rect(
        screen,
        (60, 60, 80),
        (SLIDER_X, SLIDER_Y, SLIDER_W, SLIDER_H),
        border_radius=4,
    )

    # Fill
    frac = (sim_step_ms - SLIDER_MIN_MS) / (SLIDER_MAX_MS - SLIDER_MIN_MS)
    fill_w = int(frac * SLIDER_W)
    if fill_w > 0:
        pygame.draw.rect(
            screen,
            (100, 140, 200),
            (SLIDER_X, SLIDER_Y, fill_w, SLIDER_H),
            border_radius=4,
        )

    # Handle
    handle_rect = _slider_handle_rect(sim_step_ms)
    pygame.draw.rect(screen, (180, 200, 240), handle_rect, border_radius=3)

    # Speed value label
    if sim_step_ms == 0:
        speed_val = "max"
    else:
        speed_val = f"{sim_step_ms}ms"
    val_surf = font.render(speed_val, True, (120, 120, 140))
    screen.blit(val_surf, (SLIDER_X + SLIDER_W + 8, SLIDER_Y - 2))
