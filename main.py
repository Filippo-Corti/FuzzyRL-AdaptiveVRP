import random

import config
from agent import CrispQLearningAgent, CrispQLambdaAgent
from env import VRPEnvironment, Truck
from heuristics.do_nothing import DoNothing
from heuristics.insertion import NearestInsertion
from heuristics.removal import CostliestRemoval
from heuristics.two_opt import TwoOpt
from simulation import VRPSimulation
from utils import parse_vrp_instance
from viz import run

import matplotlib

matplotlib.use("tkagg")

# random.seed(10)

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

simulation = VRPSimulation(
    environment=env,
    agent=CrispQLambdaAgent(),
    actions=[
        NearestInsertion(),
        CostliestRemoval(),
        TwoOpt(),
        DoNothing(),
    ],
)

run(simulation)
