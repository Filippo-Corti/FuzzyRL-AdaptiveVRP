import pygame

from simulation import VRPSimulation
from .plots import MetricsPlotter
from .sprites import Sprites
from .renderer import Renderer
from .hud import HUD

import config


def run(simulation: VRPSimulation):
    """
    Runs the simulation of the VRP environment with the given agent and renders it using Pygame.
    """
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
    ms_since_tick = 0

    sim = None

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
                    if not paused:
                        sim = simulation.best_snapshot
                    else:
                        sim = None
                    paused = not paused
                if event.key == pygame.K_RIGHT and paused:
                    step_once = True  # single-step while paused
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        # ── Simulation tick ──────────────────────────────────────────
        if not paused or step_once:
            ms_since_tick += dt
            if ms_since_tick >= config.SIM_STEP_MS or step_once:
                ms_since_tick = 0
                step_once = False

                simulation.step()

        # ── Render ──────────────────────────────────────────────────
        screen.fill((0, 0, 0))
        if sim is not None:
            renderer.draw(sim)
            hud.draw(sim)
        else:
            sim_state = simulation.get_snapshot()
            renderer.draw(sim_state)
            hud.draw(sim_state)
            plotter.update(sim_state)
        if tick % config.PLOT_EVERY == 0:
            plotter.draw()
        _draw_overlay_hints(screen, paused)
        pygame.display.flip()

        tick += 1


def _draw_overlay_hints(screen, paused):
    font = pygame.font.SysFont("monospace", 11)
    hints = [
        "SPACE  pause / resume",
        "->     step (when paused)",
        "ESC    quit",
    ]
    if paused:
        lbl = font.render("PAUSED", True, (240, 180, 60))
        screen.blit(lbl, (10, 10))
    for i, h in enumerate(hints):
        surf = font.render(h, True, (80, 80, 100))
        screen.blit(surf, (10, config.WINDOW_H - 18 * (len(hints) - i)))
