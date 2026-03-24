from __future__ import annotations
from typing import TYPE_CHECKING

import math
import random
import config

if TYPE_CHECKING:
    from env import VRPEnvironment
    from agent import VRPAgent
    from heuristics import HeuristicAction, Heuristic
from simulation.snapshot import SimulationSnapshot, SimulationStats
from env.truck import TruckStatus, Truck


class VRPSimulation:

    def __init__(
        self,
        environment: VRPEnvironment,
        agent: VRPAgent,
        actions: list[Heuristic],
    ):
        self.environment: VRPEnvironment = environment
        self.agent: VRPAgent = agent
        self.actions: dict[HeuristicAction, Heuristic] = {h.name(): h for h in actions}

        self.step_count: int = 0
        self.current_truck_idx: int = 0
        self.last_action: HeuristicAction | None = None
        self.last_reward: float = 0.0

        self.best_snapshot: SimulationSnapshot | None = None
        self.best_distance: float = math.inf
        self.last_snapshot: SimulationSnapshot | None = None
        self.last_distance: float = math.inf

    def step(self):
        if len(self.environment.trucks) == 0:
            return

        # Try to recover or break down the current truck
        truck = self.environment.get_truck_by_index(self.current_truck_idx)
        if truck.status == TruckStatus.BROKEN:
            self.try_recovery(truck)
        else:
            breakdown = self.try_breakdown(truck)
            self.agent.notify_of_disruption(breakdown)

        if truck.status == TruckStatus.BROKEN:
            self.next_step()
            return

        # Pick and execute action for the current truck
        self.execute_action(truck)
        self.next_step()

    def execute_action(self, truck: Truck):
        curr_obs = self.environment.get_observation(truck)
        curr_available = self.get_available_actions(truck)
        if len(curr_available) == 0:
            self.last_action = HeuristicAction.DO_NOTHING
            return

        self.last_action = self.agent.select_action(curr_obs, curr_available)
        curr_truck_distance = self.environment.compute_truck_distance(truck.id)
        self.actions[self.last_action].apply(self.environment, truck)
        new_truck_distance = self.environment.compute_truck_distance(truck.id)
        self.last_reward = self.compute_reward(
            truck, new_truck_distance - curr_truck_distance
        )

        next_obs = self.environment.get_observation(truck)
        next_available = self.get_available_actions(truck)

        self.agent.update(next_obs, self.last_reward, next_available)

    def next_step(self):
        if self.environment.graph.is_fully_assigned:
            self.last_snapshot = self.get_snapshot()
            self.last_distance = self.environment.compute_total_distance()
            if self.last_distance < self.best_distance:
                self.best_distance = self.last_distance
                self.best_snapshot = self.last_snapshot

        self.current_truck_idx = (self.current_truck_idx + 1) % len(
            self.environment.trucks
        )
        self.step_count += 1

    def get_available_actions(self, truck: Truck) -> list[HeuristicAction]:
        return [
            k
            for k, v in self.actions.items()
            if v.is_applicable(self.environment, truck)
        ]

    def compute_reward(self, truck: Truck, delta_distance: float) -> float:
        """
        Computes the reward as:
        R = -delta_distance - λ · unvisited_nodes - γ · crossings
        """
        orphans = list(self.environment.graph.unassigned_nodes())
        crossings = self.environment.count_crossings()

        lambda_weight = 2.0  # orphan penalty
        gamma_weight = 1.0  # crossing penalty

        return -delta_distance - lambda_weight * len(orphans) - gamma_weight * crossings

    def try_recovery(self, truck: Truck) -> bool:
        if random.random() < config.RECOVERY_PROB:
            self.environment.recover(truck.id)
            return True
        return False

    def try_breakdown(self, truck: Truck):
        if random.random() < config.DISRUPTION_PROB:
            self.environment.breakdown(truck.id)
            return True
        return False

    def get_snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            environment=self.environment.get_snapshot(),
            agent=self.agent.get_snapshot(),
            stats=SimulationStats(
                round=self.step_count,
                orphans=len(list(self.environment.graph.unassigned_nodes())),
                total_nodes=len(self.environment.graph.nodes),
                total_trucks=len(self.environment.trucks),
                active_trucks=sum(
                    1
                    for t in self.environment.trucks.values()
                    if t.status != TruckStatus.BROKEN
                ),
                total_distance=self.environment.compute_total_distance(),
                episode_reward=self.last_reward,
                truck_turn=self.current_truck_idx,
                last_action=self.last_action.name if self.last_action else "None",
                best_solution_distance=self.best_distance,
                last_distance=self.last_distance,
            ),
        )
