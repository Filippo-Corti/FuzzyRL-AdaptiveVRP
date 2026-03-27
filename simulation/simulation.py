from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import VRPEnvironment, EnvObservation
    from agent import VRPAgent
from simulation.snapshot import SimulationSnapshot, SimulationStats
from env.graph import VRPNode


class VRPSimulation:

    def __init__(
        self,
        environment: VRPEnvironment,
        agent: VRPAgent,
    ):
        self.environment: VRPEnvironment = environment
        self.agent = agent

        self.step_count: int = 0

    def run_episode(self, with_baseline: bool = False) -> float:
        """
        Runs a full episode of the simulation until all nodes are visited,
        returning the total reward accumulated by the agent.
        """
        total_reward = 0.0
        while not self.is_complete():
            reward = self.execute_step(record=True)
            self.next_step()
            total_reward += reward

        baseline = self.compute_baseline() if with_baseline else None
        self.agent.finish_episode(baseline)
        return total_reward

    def step(self) -> tuple[bool, float]:
        """
        During each step, the agent chooses a next node to visit
        :return: whether the simulation is done (all nodes visited)
        """
        reward = 0.0
        if self.environment.graph.orphans_count != 0:
            reward = self.execute_step()
        self.next_step()
        return self.is_complete(), reward

    def execute_step(self, record: bool = False) -> float:
        s = self.environment.get_observation()
        node_id = self.agent.select_node(s)

        if record:
            self.agent.record(s)

        if node_id == self.environment.graph.depot.id:
            self.environment.back_to_depot()
        else:
            self.environment.visit_node(self.environment.get_node_by_id(node_id))

        s_prime = self.environment.get_observation()
        reward = self.compute_reward(s, s_prime)
        self.agent.update(reward, s_prime)
        return reward

    def next_step(self):
        self.step_count += 1

    def compute_reward(self, s: EnvObservation, s_prime: EnvObservation) -> float:
        return -VRPNode.distance(s.truck_pos, s_prime.truck_pos)

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            environment=self.environment.snapshot(),
            agent=self.agent.snapshot(),
            stats=SimulationStats(
                round=self.step_count,
                orphans=self.environment.graph.orphans_count,
                total_nodes=len(self.environment.graph.nodes),
                total_distance=self.environment.compute_total_distance(),
            ),
        )

    def compute_baseline(self) -> float:
        """
        Runs the current policy in greedy mode on the current instance.
        No gradient tracking, no side effects on the agent's buffer.
        """
        self.environment.reset()
        total_reward = 0.0

        with torch.no_grad():
            while not self.is_complete():
                s = self.environment.get_observation()
                node_id = self.agent.select_node(s, greedy=True)

                if node_id == self.environment.graph.depot.id:
                    self.environment.back_to_depot()
                else:
                    self.environment.visit_node(
                        self.environment.get_node_by_id(node_id)
                    )

                s_prime = self.environment.get_observation()
                total_reward += self.compute_reward(s, s_prime)

        self.environment.reset()
        return total_reward

    def reset(self):
        """
        Resets the simulation to a clean state, preserving the environment topology.
        """
        self.step_count = 0
        self.environment.reset()

    def is_complete(self):
        return self.environment.graph.is_fully_visited and self.environment.truck.at_depot