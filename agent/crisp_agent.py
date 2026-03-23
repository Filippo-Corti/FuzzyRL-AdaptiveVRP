from __future__ import annotations
from typing import TYPE_CHECKING

from agent import VRPAgent

if TYPE_CHECKING:
    from env import EnvObservation

import random
from collections import defaultdict

import config

# Crisp bin definitions — (n_bins, min, max)
BIN_DEFINITIONS: dict[str, list[float]] = {
    "truck_load": [0.0, 0.25, 0.5, 0.75, 1.0],
    "fleet_availability": [0.0, 0.33, 0.66, 1.0],
    "orphan_pressure": [0.0, 0.25, 0.5, 0.75, 1.0],
    "nearest_orphan_dist": [0.0, 0.33, 0.66, 1.0],
}

State = tuple[int, int, int, int]
Action = str


class CrispQLearningAgent(VRPAgent):

    def __init__(self):
        super().__init__()
        self.q_table: dict[tuple[State, Action], float] = defaultdict(float)
        self.epsilon = config.EPSILON_START
        self.alpha = config.LEARNING_RATE
        self.gamma = config.DISCOUNT_FACTOR
        self.last_state: State | None = None
        self.last_action: Action | None = None

    def select_action(
        self, obs: EnvObservation, available_actions: list[Action]
    ) -> Action:
        state = self._observation_to_state(obs)

        if random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            action = self._greedy_action(state, available_actions)

        self.last_state = state
        self.last_action = action
        return action

    def update(
        self, obs: EnvObservation, reward: float, available_actions: list[Action]
    ):
        """
        Call this after the environment has transitioned as a result of the last select_action call.
        """
        if self.last_state is None or self.last_action is None:
            return

        next_state = self._observation_to_state(obs)
        best_next_q = self._max_q(next_state, available_actions)
        key = (self.last_state, self.last_action)

        self.q_table[key] += self.alpha * (
            reward + self.gamma * best_next_q - self.q_table[key]
        )

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _greedy_action(self, state: State, available_actions: list[Action]) -> Action:
        return max(available_actions, key=lambda a: self.q_table[(state, a)])

    def _max_q(self, state: State, available_actions: list[Action]) -> float:
        return max(self.q_table[(state, a)] for a in available_actions)

    @staticmethod
    def _discretise(value: float, edges: list[float]) -> int:
        for i, edge in enumerate(edges[1:], start=1):
            if value <= edge:
                return i - 1
        return len(edges) - 2

    @staticmethod
    def _observation_to_state(obs: EnvObservation) -> State:
        return (
            CrispQLearningAgent._discretise(
                obs.truck_load, BIN_DEFINITIONS["truck_load"]
            ),
            CrispQLearningAgent._discretise(
                obs.fleet_availability, BIN_DEFINITIONS["fleet_availability"]
            ),
            CrispQLearningAgent._discretise(
                obs.orphan_pressure, BIN_DEFINITIONS["orphan_pressure"]
            ),
            CrispQLearningAgent._discretise(
                obs.nearest_orphan_dist, BIN_DEFINITIONS["nearest_orphan_dist"]
            ),
        )

    def get_stats(self) -> dict[str, float]:
        return {
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
        }
