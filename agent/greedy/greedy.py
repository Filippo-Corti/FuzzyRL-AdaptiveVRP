from __future__ import annotations

from typing import TYPE_CHECKING

from agent import VRPAgent
from env import NodeObservation, VRPNode
from simulation.snapshot import AgentSnapshot

if TYPE_CHECKING:
    from env import EnvObservation


class GreedyAgent(VRPAgent):
    """
    A simple agent that always selects the closest unvisited node with demand, ignoring future consequences.
    """

    def __init__(self):
        super().__init__()
        self.last_choice: NodeObservation | None = None

    def select_node(self, obs: EnvObservation, greedy: bool = True) -> int:
        """
        Selects the id of the closest unvisited node with demand.
        """
        truck_x, truck_y = obs.truck_pos
        best_node = None
        best_dist = float("inf")

        customer_nodes = [
            node
            for node in obs.nodes
            if node.demand > 0
            and not node.visited
            and node.demand + obs.truck_load <= obs.truck_capacity
        ]

        available_nodes = (
            customer_nodes
            if customer_nodes
            else [node for node in obs.nodes if node.depot]
        )

        for node in available_nodes:
            dist = VRPNode.distance((truck_x, truck_y), (node.x, node.y))
            if dist < best_dist:
                best_dist = dist
                best_node = node

        if best_node:
            self.last_choice = best_node
            return best_node.id
        assert False

    def record(self, obs: EnvObservation):
        pass

    def update(self, reward: float, obs: EnvObservation):
        pass

    def finish_episode(self, baseline: float | None = None):
        self.last_choice = None

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            last_choice=(
                (self.last_choice.x, self.last_choice.y) if self.last_choice else None
            ),
            epsilon=None,
        )
