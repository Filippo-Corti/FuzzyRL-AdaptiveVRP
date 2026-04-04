import torch
from agent.transformer_agent import TransformerAgent
from env import VRPEnvironment, Truck
from simulation import VRPSimulation
from utils import generate_vrp_instance
from viz import run
import config


def create_simulation(agent: TransformerAgent) -> VRPSimulation:
    graph, capacity = generate_vrp_instance(num_nodes=config.NUM_NODES)
    depot_x, depot_y = graph.depot.pos
    truck = Truck(id=0, pos=(depot_x, depot_y), capacity=capacity)
    env = VRPEnvironment(graph=graph, truck=truck)
    return VRPSimulation(environment=env, agent=agent)


def load_agent(path: str, d_model: int = 128) -> TransformerAgent:
    agent = TransformerAgent(node_features=4, state_features=3, d_model=d_model)
    checkpoint = torch.load(path, map_location="cpu")
    agent.encoder.load_state_dict(checkpoint["encoder"])
    agent.decoder.load_state_dict(checkpoint["decoder"])
    agent.encoder.eval()
    agent.decoder.eval()
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    return agent


if __name__ == "__main__":
    agent = load_agent("checkpoints/transformer.pt")
    simulation = create_simulation(agent)
    run(simulation, lambda: create_simulation(agent))
