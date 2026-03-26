import matplotlib

matplotlib.use("tkagg")

import config
from agent import GreedyAgent
from env import VRPEnvironment, Truck
from simulation import VRPSimulation
from utils import generate_vrp_instance
from viz import run


def create_simulation():
    # Generate new instance
    graph, vehicle_capacity = generate_vrp_instance(num_nodes=config.NUM_NODES)

    depot_x, depot_y = graph.depot.pos

    truck = Truck(
        id=0,
        pos=(depot_x, depot_y),
        capacity=vehicle_capacity,
    )

    env = VRPEnvironment(
        graph=graph,
        truck=truck,
    )

    agent = GreedyAgent()

    return VRPSimulation(environment=env, agent=agent)


if __name__ == "__main__":
    simulation = create_simulation()
    run(simulation, create_simulation)
