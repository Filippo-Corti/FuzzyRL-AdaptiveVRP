import random

import config
from env import VRPEnvironment, Truck
from utils import parse_vrp_instance
from viz import run

# random.seed(10)

graph = parse_vrp_instance(path="assets/datasets/CVRPLIB-Augerat-A/A-n32-k5.vrp")

env = VRPEnvironment(graph=graph)

depot_x, depot_y = graph.depot.pos

for i in range(config.NUM_TRUCKS):
    env.add_truck(
        Truck(
            id=i,
            pos=(depot_x, depot_y + 0.05 * (i + 1)),
            capacity=config.TRUCK_CAPACITY,
        )
    )
run(env, None)  # blocks until window is closed
