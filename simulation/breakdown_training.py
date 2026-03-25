from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from env import VRPEnvironment
    from agent import VRPAgent
    from heuristics import HeuristicAction, Heuristic

MAX_STEPS = 100
LAMBDA_ORPHAN = 10.0
GAMMA_CROSSINGS = 0.5
TERMINAL_BONUS = 40.0
TERMINAL_DISTANCE_PENALTY = 4.0
TERMINAL_CROSSING_PENALTY = 1.0
TIMEOUT_PENALTY = -20.0


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


class BreakdownTraining:

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
            delta_distance = self.apply_action(action, truck)
            reward = self.per_step_reward(delta_distance)
            total_reward += reward

            done = self.environment.graph.is_fully_assigned
            is_last_step = step == MAX_STEPS - 1

            if done or is_last_step:
                terminal = self.terminal_reward(success=done)
                total_reward += terminal
                next_obs = self.environment.get_observation(truck)
                next_available = self.available_actions(truck)
                self.agent.update(next_obs, reward + terminal, next_available)
                return EpisodeResult(
                    number=0,
                    steps=step + 1,
                    success=done,
                    final_distance=self.environment.compute_total_distance(),
                    final_crossings=self.environment.count_crossings(),
                    final_orphans=len(list(self.environment.graph.unassigned_nodes())),
                    total_reward=total_reward,
                    epsilon=self.agent.epsilon,
                    q_table_size=len(self.agent.q_table),
                )

            next_obs = self.environment.get_observation(truck)
            next_available = self.available_actions(truck)
            self.agent.update(next_obs, reward, next_available)
            self.advance_truck()

        assert False  # Should never reach here

    def train(self, n_episodes: int):
        for episode in range(n_episodes):
            result = self.run_episode()
            result.number = episode + 1
            if self.on_episode_end:
                self.on_episode_end(result)
            if (episode + 1) % 10 == 0:
                status = "SUCCESS" if result.success else "TIMEOUT"
                print(
                    f"[{status}] Episode {episode + 1}/{n_episodes} | "
                    f"steps={result.steps:3d} | "
                    f"orphans={result.final_orphans} | "
                    f"dist={result.final_distance:.2f} | "
                    f"reward={result.total_reward:.1f} | "
                    f"eps={result.epsilon:.4f} | "
                    f"Q={result.q_table_size}"
                )

    def reset_episode(self):
        # Clear all routes and unassign all nodes
        for truck in self.environment.trucks.values():
            truck.route.clear()
            truck.recover()
        for node in self.environment.graph:
            node.assignment = None

        # Insert in random truck order to diversify starting states
        truck_list = list(self.environment.trucks.values())
        random.shuffle(truck_list)
        for truck in truck_list:
            while self.insert_heuristic.is_applicable(self.environment, truck):
                self.insert_heuristic.apply(self.environment, truck)

        # Break down a random truck
        truck = random.choice(list(self.environment.trucks.values()))
        self.environment.breakdown(truck.id)

        # Reset episode counters
        self.current_truck_idx = 0
        self.agent.notify_of_disruption()

    def current_active_truck(self):
        active = [t for t in self.environment.trucks.values() if t.is_active()]
        # Round-robin over active trucks only
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

    def apply_action(self, action: HeuristicAction, truck) -> float:
        before = self.environment.compute_truck_distance(truck.id)
        self.actions[action].apply(self.environment, truck)
        after = self.environment.compute_truck_distance(truck.id)
        return after - before

    def per_step_reward(self, delta_distance: float) -> float:
        orphans = len(list(self.environment.graph.unassigned_nodes()))
        crossings = self.environment.count_crossings()
        return -delta_distance - LAMBDA_ORPHAN * orphans * -GAMMA_CROSSINGS * crossings

    def terminal_reward(self, success: bool) -> float:
        if not success:
            return TIMEOUT_PENALTY
        distance = self.environment.compute_total_distance()
        crossings = self.environment.count_crossings()
        return (
            TERMINAL_BONUS
            - TERMINAL_DISTANCE_PENALTY * distance
            - TERMINAL_CROSSING_PENALTY * crossings
        )
