import pygame

from simulation import VRPSimulation
from .plots import MetricsPlotter
from .sprites import Sprites
from .renderer import SimulationUI
from .hud import HUD

import config

SLIDER_X = 10
SLIDER_Y = 40
SLIDER_W = 180
SLIDER_H = 8
SLIDER_MIN_MS = 0
SLIDER_MAX_MS = 500


def run(simulation: VRPSimulation, simulation_factory):
    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
    pygame.display.set_caption("VRP Reinforcement Learning")
    clock = pygame.time.Clock()
    Sprites.init()

    graph_rect = pygame.Rect(0, 0, config.GRAPH_W, config.WINDOW_H)
    hud_rect = pygame.Rect(config.GRAPH_W, 0, config.HUD_W, config.WINDOW_H)

    renderer = SimulationUI(screen, graph_rect)
    hud = HUD(screen, hud_rect)

    paused = False
    step_once = False
    episodes = 0
    ms_since_tick = 0
    sim_step_ms = config.SIM_STEP_MS
    dragging_slider = False
    done = False

    tick = 0

    while True:
        dt = clock.tick(config.FPS_CAP)

        # --- EVENTS ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT and paused:
                    step_once = True
                if event.key == pygame.K_r:
                    simulation = simulation_factory()  # manual reset
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

        # --- SIMULATION ---
        if not paused or step_once:
            ms_since_tick += dt

            if done:
                R = sum(simulation.agent.rewards)
                baseline = simulation.compute_baseline()
                simulation.agent.finish_episode(baseline=baseline)
                print(f"Episode {episodes} completed (total distance = {R}).")
                simulation = simulation_factory()
                done = False
                episodes += 1
            elif ms_since_tick >= sim_step_ms or step_once:
                ms_since_tick = 0
                step_once = False

                reward = simulation.execute_step(record=True)
                simulation.next_step()
                done = simulation.is_complete()

        # --- RENDER ---
        if episodes > 1000:
            screen.fill((0, 0, 0))

            snapshot = simulation.snapshot()

            renderer.draw(snapshot)
            hud.draw(snapshot)

            _draw_controls(screen, paused, sim_step_ms)
            pygame.display.flip()

        tick += 1


def _slider_handle_rect(sim_step_ms: int) -> pygame.Rect:
    frac = (sim_step_ms - SLIDER_MIN_MS) / (SLIDER_MAX_MS - SLIDER_MIN_MS)
    handle_x = int(SLIDER_X + frac * SLIDER_W)
    return pygame.Rect(handle_x - 6, SLIDER_Y - 6, 12, SLIDER_H + 12)


def _draw_controls(screen, paused: bool, sim_step_ms: int):
    font = pygame.font.SysFont("monospace", 11)

    # Hints
    hints = [
        "SPACE  pause / resume",
        "->     step (when paused)",
        "ESC    quit",
    ]
    for i, h in enumerate(hints):
        color = (80, 80, 100)
        surf = font.render(h, True, color)
        screen.blit(surf, (10, config.WINDOW_H - 18 * (len(hints) - i)))

    # Paused indicator
    if paused:
        lbl = font.render("PAUSED", True, (240, 180, 60))
        screen.blit(lbl, (10, 10))

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
