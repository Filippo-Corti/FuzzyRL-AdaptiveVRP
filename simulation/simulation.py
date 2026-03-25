from __future__ import annotations

from enum import StrEnum
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


class SimulationMode(StrEnum):
    IDLE = "Idle"
    RECOVERING = "Recovering"
    RECOVERED = "Recovered"
    REBALANCING = "Rebalancing"


class VRPSimulation:

    REBALANCING_PATIENCE = 1000  # consecutive steps with negligible reward improvement
    REBALANCING_IMBALANCE_THRESHOLD = 0.15  # normalised std dev considered acceptable
    IMPROVEMENT_THRESHOLD = 1e-3

    def __init__(
        self,
        environment: VRPEnvironment,
        breakdown_agent: VRPAgent,
        rebalancing_agent: VRPAgent,
        breakdown_actions: list[Heuristic],
        rebalancing_actions: list[Heuristic],
    ):
        self.environment: VRPEnvironment = environment
        self.breakdown_agent: VRPAgent = breakdown_agent
        self.rebalancing_agent: VRPAgent = rebalancing_agent
        self.breakdown_actions: dict[HeuristicAction, Heuristic] = {
            h.name(): h for h in breakdown_actions
        }
        self.rebalancing_actions: dict[HeuristicAction, Heuristic] = {
            h.name(): h for h in rebalancing_actions
        }

        self.mode: SimulationMode = SimulationMode.IDLE
        self.step_count: int = 0
        self.fully_assigned_steps: int = 0
        self.no_improvement_steps: int = 0
        self.last_rebalancing_reward: float = 0
        self.current_truck_idx: int = 0
        self.last_action: HeuristicAction | None = None
        self.last_reward: float = 0.0

        self.best_snapshot: SimulationSnapshot | None = None
        self.best_distance: float = math.inf
        self.last_snapshot: SimulationSnapshot | None = None
        self.last_distance: float = math.inf

    @property
    def active_agent(self) -> VRPAgent:
        if self.mode == SimulationMode.RECOVERING:
            return self.breakdown_agent
        return (
            self.rebalancing_agent
        )  # Not perfect, as we treat both recovered and rebalancing as "recovered" for agent purposes

    @property
    def active_actions(self) -> dict[HeuristicAction, Heuristic]:
        if self.mode == SimulationMode.RECOVERING:
            return self.breakdown_actions
        return self.rebalancing_actions

    @property
    def steps_until_stable(self) -> int:
        return max(4, len(self.environment.trucks))

    def set_mode(self, mode: SimulationMode):
        if mode != self.mode:
            self.fully_assigned_steps = 0
            self.no_improvement_steps = 0
            self.last_rebalancing_reward = 0
        self.mode = mode

    def initialize_environment(self, insert_heuristic: Heuristic):
        """
        Initializes the environment by inserting nodes using the given heuristic.
        """
        for truck in self.environment.trucks.values():
            while insert_heuristic.is_applicable(self.environment, truck):
                insert_heuristic.apply(self.environment, truck)

    def step(self):
        if len(self.environment.trucks) == 0:
            return

        truck = self.environment.get_truck_by_index(self.current_truck_idx)
        if truck.status == TruckStatus.BROKEN:
            recovered = self.try_recovery(truck)
            if not recovered:
                self.next_step()
                return
        else:
            breakdown = self.try_breakdown(truck)
            if breakdown:
                self.active_agent.notify_of_disruption()
                self.next_step()
                return

        if self.mode in (SimulationMode.RECOVERING, SimulationMode.REBALANCING):
            self.execute_action(truck)

        self.next_step()

    def execute_action(self, truck: Truck):
        agent = self.active_agent
        actions = self.active_actions

        curr_obs = self.environment.get_observation(truck)
        curr_available = self.get_available_actions(truck, actions)
        if len(curr_available) == 0:
            self.last_action = HeuristicAction.DO_NOTHING
            return

        self.last_action = agent.select_action(curr_obs, curr_available)
        curr_truck_distance = self.environment.compute_truck_distance(truck.id)
        actions[self.last_action].apply(self.environment, truck)
        new_truck_distance = self.environment.compute_truck_distance(truck.id)
        self.last_reward = self.compute_reward(
            truck, new_truck_distance - curr_truck_distance
        )

        if self.mode == SimulationMode.REBALANCING:
            if (
                abs(self.last_reward - self.last_rebalancing_reward)
                < self.IMPROVEMENT_THRESHOLD
            ):
                self.no_improvement_steps += 1
            else:
                self.no_improvement_steps = 0
            self.last_rebalancing_reward = self.last_reward

        next_obs = self.environment.get_observation(truck)
        next_available = self.get_available_actions(truck, actions)
        agent.update(next_obs, self.last_reward, next_available)

    def next_step(self):
        active_trucks = [
            t
            for t in self.environment.trucks.values()
            if t.status != TruckStatus.BROKEN
        ]
        if len(active_trucks) == 0:
            self.current_truck_idx = (self.current_truck_idx + 1) % len(
                self.environment.trucks
            )
            self.step_count += 1
            return

        if self.environment.graph.is_fully_assigned:
            self.fully_assigned_steps += 1
            if self.fully_assigned_steps >= self.steps_until_stable:
                if self.mode == SimulationMode.RECOVERING:
                    self.set_mode(SimulationMode.RECOVERED)
            if self.mode == SimulationMode.REBALANCING:
                if self.rebalancing_has_stabilised():
                    self.set_mode(SimulationMode.IDLE)

            self.last_snapshot = self.get_snapshot()
            self.last_distance = self.environment.compute_total_distance()
            if self.last_distance < self.best_distance:
                self.best_distance = self.last_distance
                self.best_snapshot = self.last_snapshot
        else:
            self.fully_assigned_steps = 0

        self.current_truck_idx = (self.current_truck_idx + 1) % len(
            self.environment.trucks
        )
        self.step_count += 1

    def get_available_actions(
        self, truck: Truck, actions: dict[HeuristicAction, Heuristic]
    ) -> list[HeuristicAction]:
        return [
            k for k, v in actions.items() if v.is_applicable(self.environment, truck)
        ]

    def compute_reward(self, truck: Truck, delta_distance: float) -> float:
        match self.mode:
            case SimulationMode.RECOVERING:
                return self.breakdown_reward(delta_distance)
            case SimulationMode.REBALANCING:
                return self.rebalancing_reward(delta_distance)
            case SimulationMode.IDLE | SimulationMode.RECOVERED:
                return 0.0

    def breakdown_reward(self, delta_distance: float) -> float:
        orphans = len(list(self.environment.graph.unassigned_nodes()))
        crossings = self.environment.count_crossings()
        return -delta_distance - 1.0 * orphans - 0.1 * crossings

    def rebalancing_reward(self, delta_distance: float) -> float:
        imbalance = self.environment.compute_imbalance()
        crossings = self.environment.count_crossings()
        return -delta_distance - 1.0 * imbalance - 0.3 * crossings

    def try_recovery(self, truck: Truck) -> bool:
        if (
            random.random() < config.RECOVERY_PROB
            and self.mode == SimulationMode.RECOVERED  # only fires from RECOVERED
        ):
            self.environment.recover(truck.id)
            self.set_mode(SimulationMode.REBALANCING)
            return True
        return False

    def try_breakdown(self, truck: Truck) -> bool:
        if (
            random.random() < config.DISRUPTION_PROB
            and self.mode == SimulationMode.IDLE  # only fires from IDLE
        ):
            self.environment.breakdown(truck.id)
            self.set_mode(SimulationMode.RECOVERING)
            return True
        return False

    def rebalancing_has_stabilised(self) -> bool:
        imbalance_ok = (
            self.environment.compute_imbalance() < self.REBALANCING_IMBALANCE_THRESHOLD
        )
        patience_exhausted = self.no_improvement_steps >= self.REBALANCING_PATIENCE
        return imbalance_ok or patience_exhausted

    def get_snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            environment=self.environment.get_snapshot(),
            agent=self.active_agent.get_snapshot(),
            stats=SimulationStats(
                round=self.step_count,
                status=str(self.mode),
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

    def reset(self):
        """Resets the simulation to a clean state, preserving the environment topology."""
        # Reset all trucks
        for truck in self.environment.trucks.values():
            truck.recover()  # bring back any broken trucks
            truck.route.clear()  # empty all routes

        # Unassign all nodes
        for node in self.environment.graph:
            node.assignment = None

        # Reset simulation state
        self.set_mode(SimulationMode.IDLE)
        self.step_count = 0
        self.fully_assigned_steps = 0
        self.no_improvement_steps = 0
        self.current_truck_idx = 0
        self.last_action = None
        self.last_reward = 0.0
        self.last_rebalancing_reward = 0.0
