import matplotlib

matplotlib.use("tkagg")

import config
from agent import VRPAgent, greedy, transformer
from env import VRPEnvironment, Truck
from simulation import VRPSimulation
from utils import generate_vrp_instance
from viz import run


def create_simulation(agent: VRPAgent | None = None) -> VRPSimulation:
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

    if not agent:
        agent = greedy.GreedyAgent()

    return VRPSimulation(environment=env, agent=agent)


if __name__ == "__main__":
    agent = transformer.TransformerAgent(
        nodes_features=4, state_features=3, d_model=128
    )
    simulation = create_simulation(agent)
    run(simulation, lambda: create_simulation(agent))
