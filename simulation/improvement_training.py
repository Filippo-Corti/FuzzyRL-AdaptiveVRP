from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from env import VRPEnvironment
    from agent import VRPAgent
    from heuristics import HeuristicAction, Heuristic

MAX_STEPS = 100
TERMINAL_BONUS = 0.0
TERMINAL_DISTANCE_PENALTY = 1.0
TERMINAL_CROSSING_PENALTY = 5.0
TIMEOUT_PENALTY = -50.0


@dataclass
class EpisodeResult:
    number: int
    steps: int
    success: bool
    final_distance: float
    final_crossings: int
    final_orphans: int
    total_reward: float
    epsilon: float
    q_table_size: int


class ImprovementTraining:

    def __init__(
        self,
        environment: VRPEnvironment,
        agent: VRPAgent,
        actions: list[Heuristic],
        insert_heuristic: Heuristic,
        on_episode_end: Callable[[EpisodeResult], None] | None = None,
    ):
        self.environment = environment
        self.agent = agent
        self.actions: dict[HeuristicAction, Heuristic] = {h.name(): h for h in actions}
        self.insert_heuristic = insert_heuristic
        self.current_truck_idx = 0
        self.on_episode_end = on_episode_end

    def run_episode(self) -> EpisodeResult:
        self.reset_episode()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            truck = self.current_active_truck()
            obs = self.environment.get_observation(truck)
            available = self.available_actions(truck)

            action = self.agent.select_action(obs, available)
            delta_distance, delta_crossings = self.apply_action(action, truck)
            reward = self.per_step_reward(delta_distance, delta_crossings)
            total_reward += reward

            # Episode always runs full MAX_STEPS
            done = step == MAX_STEPS - 1

            next_obs = self.environment.get_observation(truck)
            next_available = self.available_actions(truck)
            self.agent.update(next_obs, reward, next_available)
            self.advance_truck()

            if done:
                terminal = self.terminal_reward()
                total_reward += terminal
                self.agent.update(next_obs, terminal, next_available)
                return EpisodeResult(
                    number=0,
                    steps=step + 1,
                    success=True,  # always feasible
                    final_distance=self.environment.compute_total_distance(),
                    final_crossings=self.environment.count_crossings(),
                    final_orphans=0,
                    total_reward=total_reward,
                    epsilon=self.agent.epsilon,
                    q_table_size=len(self.agent.q_table),
                )

        assert False  # Should never reach here

    def train(self, n_episodes: int):
        for episode in range(n_episodes):
            result = self.run_episode()
            result.number = episode + 1
            if self.on_episode_end:
                self.on_episode_end(result)
            if (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"steps={result.steps:3d} | "
                    f"dist={result.final_distance:.2f} | "
                    f"crossings={result.final_crossings} | "
                    f"reward={result.total_reward:.1f} | "
                    f"eps={result.epsilon:.4f} | "
                    f"Q={result.q_table_size}"
                )

    def reset_episode(self):
        # Reset truck routes and assignments to a feasible solution
        for truck in self.environment.trucks.values():
            truck.route.clear()
            truck.recover()
        for node in self.environment.graph:
            node.assignment = None

        # Generate initial feasible solution with insert heuristic
        unfilled_trucks = [t for t in self.environment.trucks.values() if not t.is_full]
        while not self.environment.graph.is_fully_assigned:
            # Pick a random truck that still has capacity
            truck = random.choice(unfilled_trucks)

            if self.insert_heuristic.is_applicable(self.environment, truck):
                self.insert_heuristic.apply(self.environment, truck)
            else:
                print(
                    f"Nopers for some reason ({self.environment.graph.orphans_count})"
                )

            # Update the list of trucks with remaining capacity
            unfilled_trucks = [
                t for t in self.environment.trucks.values() if not t.is_full
            ]

        self.current_truck_idx = 0
        self.agent.notify_of_disruption()  # can use to reset internal state

    def current_active_truck(self):
        active = [t for t in self.environment.trucks.values() if t.is_active()]
        idx = self.current_truck_idx % len(active)
        return active[idx]

    def advance_truck(self):
        self.current_truck_idx += 1

    def available_actions(self, truck) -> list[HeuristicAction]:
        return [
            k
            for k, v in self.actions.items()
            if v.is_applicable(self.environment, truck)
        ]

    def apply_action(self, action: HeuristicAction, truck) -> tuple[float, int]:
        before_distance = self.environment.compute_truck_distance(truck.id)
        before_crossings = self.environment.count_crossings()
        self.actions[action].apply(self.environment, truck)
        after_distance = self.environment.compute_truck_distance(truck.id)
        after_crossings = self.environment.count_crossings()
        return after_distance - before_distance, after_crossings - before_crossings

    def per_step_reward(self, delta_distance: float, delta_crossings: float) -> float:
        return -delta_distance - 5.0 * (
            delta_crossings**2
        )  # stepwise improvement reward

    def terminal_reward(self) -> float:
        distance = self.environment.compute_total_distance()
        crossings = self.environment.count_crossings()
        return (
            TERMINAL_BONUS
            - TERMINAL_DISTANCE_PENALTY * distance
            - TERMINAL_CROSSING_PENALTY * crossings
        )
