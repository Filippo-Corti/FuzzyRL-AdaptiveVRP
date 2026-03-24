from __future__ import annotations
from typing import TYPE_CHECKING

import config
import random
from collections import defaultdict

if TYPE_CHECKING:
    from env import EnvObservation
    from heuristics import HeuristicAction
from simulation.snapshot import AgentSnapshot
from agent import VRPAgent

BIN_DEFINITIONS: dict[str, list[float]] = {
    "truck_load": [0.0, 0.25, 0.5, 0.75, 1.0],
    "fleet_availability": [0.0, 0.4, 0.6, 0.8, 1.0],
    "orphan_pressure": [0.0, 0.1, 0.25, 0.5, 1.0],
    "nearest_orphan_dist": [0.0, 0.08, 0.20, 0.40, 1.0],
    "nearest_orphan_rel_dist": [0.0, 0.4, 0.7, 1.0, 1.5, 2.0],
    "route_efficiency": [0.0, 0.3, 0.6, 0.9, 1.5, 2.0],
}

State = tuple[int, int, int, int, int, int]


class CrispQLambdaAgent(VRPAgent):

    def __init__(self):
        super().__init__()
        self.q_table: dict[tuple[State, HeuristicAction], float] = defaultdict(float)
        self.traces: dict[tuple[State, HeuristicAction], float] = defaultdict(float)

        self.epsilon = config.EPSILON_START
        self.alpha = config.LEARNING_RATE
        self.gamma = config.DISCOUNT_FACTOR
        self.lam = config.LAMBDA  # eligibility trace decay
        self.decay_frequency = config.DECAY_EVERY

        self.last_state: State | None = None
        self.last_action: HeuristicAction | None = None
        self.updates_since_decay = 0
        self.disruption_flag = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(
        self, obs: EnvObservation, available_actions: list[HeuristicAction]
    ) -> HeuristicAction:
        state = self.observation_to_state(obs)

        if random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            action = self.greedy_action(state, available_actions)

        self.last_state = state
        self.last_action = action
        return action

    def update(
        self,
        next_obs: EnvObservation,
        reward: float,
        available_actions: list[HeuristicAction],
    ):
        assert self.last_state is not None and self.last_action is not None

        next_state = self.observation_to_state(next_obs)
        best_next_q = self.max_q(next_state, available_actions)

        # TD error — same as standard Q-learning
        td_error = (
            reward
            + self.gamma * best_next_q
            - self.q_table[(self.last_state, self.last_action)]
        )

        # Accumulate trace for the current state-action pair
        self.traces[(self.last_state, self.last_action)] += 1.0

        # Update ALL entries in Q and traces proportionally to their trace value
        for key in list(self.traces.keys()):
            self.q_table[key] += self.alpha * td_error * self.traces[key]
            self.traces[key] *= self.gamma * self.lam

            # Prune negligible traces to keep memory bounded
            if self.traces[key] < 1e-6:
                del self.traces[key]

        # Reset traces on disruption
        if self.disruption_flag:
            self.reset_traces()

        # Epsilon decay
        if self.updates_since_decay >= self.decay_frequency:
            self.decay_epsilon()
            self.updates_since_decay = 0
        else:
            self.updates_since_decay += 1

    def reset_traces(self):
        """Call this when a disruption fires — past decisions are less relevant."""
        self.traces.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

    def greedy_action(
        self, state: State, available_actions: list[HeuristicAction]
    ) -> HeuristicAction:
        return max(available_actions, key=lambda a: self.q_table[(state, a)])

    def max_q(self, state: State, available_actions: list[HeuristicAction]) -> float:
        return max(self.q_table[(state, a)] for a in available_actions)

    def notify_of_disruption(self, disruption: bool):
        self.disruption_flag = disruption

    @staticmethod
    def discretize(value: float, edges: list[float]) -> int:
        for i, edge in enumerate(edges[1:], start=1):
            if value <= edge:
                return i - 1
        return len(edges) - 2

    @staticmethod
    def observation_to_state(obs: EnvObservation) -> State:
        return (
            CrispQLambdaAgent.discretize(obs.truck_load, BIN_DEFINITIONS["truck_load"]),
            CrispQLambdaAgent.discretize(
                obs.fleet_availability, BIN_DEFINITIONS["fleet_availability"]
            ),
            CrispQLambdaAgent.discretize(
                obs.orphan_pressure, BIN_DEFINITIONS["orphan_pressure"]
            ),
            CrispQLambdaAgent.discretize(
                obs.nearest_orphan_dist, BIN_DEFINITIONS["nearest_orphan_dist"]
            ),
            CrispQLambdaAgent.discretize(
                obs.nearest_orphan_rel_dist, BIN_DEFINITIONS["nearest_orphan_rel_dist"]
            ),
            CrispQLambdaAgent.discretize(
                obs.route_efficiency, BIN_DEFINITIONS["route_efficiency"]
            ),
        )

    def get_snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            memberships={},
            q_values=defaultdict(float),
            chosen_action=str(self.last_action),
            q_table_size=len(self.q_table),
            epsilon=self.epsilon,
        )

    def print_q_table(self):
        print("--------------------------------")
        for (state, action), q in self.q_table.items():
            state_str = ", ".join(
                f"{list(BIN_DEFINITIONS.keys())[i]}: {s}" for i, s in enumerate(state)
            )
            print(f"State: {state_str}, Action: {action}, Q-value: {q:.2f}")
        print("--------------------------------")
