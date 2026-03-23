import pygame

from agent import VRPAgent
from env import VRPEnvironment
from .plots import MetricsPlotter
from .sprites import Sprites
from .renderer import Renderer
from .hud import HUD

import config


def run(env: VRPEnvironment, agent: VRPAgent):
    """
    env   — your environment object (must expose get_render_state and get_stats)
    agent — your agent object (must expose step and get_render_info)
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
                        sim = env.best_snapshot
                    else:
                        sim = None
                    paused = not paused
                if event.key == pygame.K_RIGHT and paused:
                    step_once = True  # single-step while paused
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

        # ── Simulation tick ─────────────────────────────────────────
        if not paused or step_once:
            ms_since_tick += dt
            if ms_since_tick >= config.SIM_STEP_MS or step_once:
                ms_since_tick = 0
                step_once = False

                # Run one full round: all active trucks act, then disruptions fire
                env.step(agent)
                # env.step() calls agent.select_action() internally for each truck
                # and returns the last truck's agent_info dict

        # ── Render ──────────────────────────────────────────────────
        screen.fill((0, 0, 0))
        if sim is not None:
            renderer.draw(sim)
            hud.draw(sim)
        else:
            sim_state = env.get_render_state()
            renderer.draw(sim_state)
            hud.draw(sim_state)
            agent_stats = agent.get_stats()
            plotter.update(
                sim_state, agent_stats["epsilon"], int(agent_stats["q_table_size"])
            )
        if tick % 5000 == 0:
            plotter.draw()
            print(agent.q_table)
        _draw_overlay_hints(screen, paused)
        pygame.display.flip()

        if env.steps % config.DECAY_EVERY == 0:
            agent.decay_epsilon()

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
