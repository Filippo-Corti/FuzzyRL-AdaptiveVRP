import random

import config
from agent import VRPAgent
from env import VRPEnvironment, Truck
from heuristics.do_nothing import DoNothing
from heuristics.insertion import NearestInsertion
from heuristics.removal import CostliestRemoval
from heuristics.two_opt import TwoOpt
from utils import parse_vrp_instance
from viz import run

# random.seed(10)

graph = parse_vrp_instance(path="assets/datasets/CVRPLIB-Augerat-A/A-n32-k5.vrp")

env = VRPEnvironment(
    graph=graph,
    heuristics=[DoNothing, NearestInsertion, CostliestRemoval, TwoOpt],
)

depot_x, depot_y = graph.depot.pos

print(f"Capacity: {config.TRUCK_CAPACITY}")

for i in range(config.NUM_TRUCKS):
    env.add_truck(
        Truck(
            id=i,
            pos=(depot_x, depot_y + 0.04 * (i + 1)),
            capacity=config.TRUCK_CAPACITY,
        )
    )

agent = VRPAgent()

run(env, agent)
