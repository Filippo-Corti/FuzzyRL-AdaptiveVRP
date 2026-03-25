import random

import config
from agent import (
    CrispQLearningAgent,
    CrispQLambdaAgent,
    BreakdownAgent,
    RebalancingAgent,
)
from env import VRPEnvironment, Truck
from heuristics import *
from simulation import VRPSimulation, SimulationMode
from train import train_breakdown_agent, train_rebalancing_agent
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

breakdown_agent = BreakdownAgent()
rebalancing_agent = RebalancingAgent()

breakdown_actions = [DoNothing(), NearestInsertion(), CostliestRemoval(), TwoOpt()]
rebalancing_actions = [DoNothing(), CrossInsert(), TwoOpt()]

simulation = VRPSimulation(
    environment=env,
    breakdown_agent=breakdown_agent,
    rebalancing_agent=rebalancing_agent,
    breakdown_actions=breakdown_actions,
    rebalancing_actions=rebalancing_actions,
)

# Phase 1: train breakdown agent alone
train_breakdown_agent(simulation, NearestInsertion(), n_episodes=1000)

# Phase 2: train rebalancing agent alone
# (breakdown agent is now trained and used as the bridge in step 3) <-- Wrong?
train_rebalancing_agent(simulation, NearestInsertion(), n_episodes=1000)

simulation.reset()
simulation.initialize_environment(NearestInsertion())
run(simulation)
