from env import VRPEnvironment, Truck
from agent.breakdown_agent import BreakdownAgent
from heuristics import NearestInsertion, CostliestRemoval, TwoOpt
from simulation import BreakdownTraining
from simulation.breakdown_training import EpisodeResult
from utils import parse_vrp_instance

import config
import pygame
import matplotlib


from viz.renderer import Renderer
from viz.train_plots import BreakdownTrainingPlotter
from viz.visual_episode import VisualEpisode

matplotlib.use("tkagg")

graph = parse_vrp_instance(path="assets/datasets/CVRPLIB-Augerat-A/A-n32-k5.vrp")
depot_x, depot_y = graph.depot.pos

trucks = [
    Truck(id=i, pos=(depot_x, depot_y + 0.04 * (i + 1)), capacity=config.TRUCK_CAPACITY)
    for i in range(config.NUM_TRUCKS)
]

env = VRPEnvironment(
    graph=graph,
    trucks=trucks,
)
agent = BreakdownAgent()

plotter = BreakdownTrainingPlotter()


def on_episode_end(result: EpisodeResult):
    plotter.record(result)
    if (result.number + 1) % 100 == 0:
        plotter.draw()


training = BreakdownTraining(
    environment=env,
    agent=agent,
    actions=[NearestInsertion(), CostliestRemoval(), TwoOpt()],
    insert_heuristic=NearestInsertion(),
    on_episode_end=on_episode_end,
)

training.train(n_episodes=8000)

# --- visualisation ---
pygame.init()
screen = pygame.display.set_mode((config.WINDOW_W, config.WINDOW_H))
pygame.display.set_caption("Breakdown recovery — trained agent")
graph_rect = pygame.Rect(0, 0, config.WINDOW_W, config.WINDOW_H)

renderer = Renderer(surface=screen, graph_rect=graph_rect)

visual = VisualEpisode(
    training=training,
    renderer=renderer,
    step_delay=0.4,  # slow enough to follow
)
visual.run_n(n=5)  # watch 5 episodes

pygame.quit()
