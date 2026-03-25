from __future__ import annotations
from typing import TYPE_CHECKING

import config
from agent.crisp_q_lambda import CrispQLambdaAgent

if TYPE_CHECKING:
    from env import EnvObservation

State = tuple[int, int, int, int, int, int]

BIN_DEFINITIONS: dict[str, list[float]] = {
    "orphan_pressure": [0.0, 0.1, 0.25, 0.5, 1.0],
    "nearest_orphan_dist": [0.0, 0.08, 0.20, 0.40, 1.0],
    "nearest_orphan_rel_dist": [0.0, 0.25, 0.5, 0.75, 1.0],
    "truck_load": [0.0, 0.25, 0.5, 0.75, 1.0],
    "fleet_availability": [0.0, 0.4, 0.6, 0.8, 1.0],
    "insertion_cost": [0.0, 0.15, 0.3, 0.5, 1.0],  # shifted left
}


class BreakdownAgent(CrispQLambdaAgent):

    def decay_epsilon(self):
        self.epsilon = max(
            config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY_RECOVERING
        )

    @staticmethod
    def observation_to_state(obs: EnvObservation) -> State:
        return (
            CrispQLambdaAgent.discretize(
                obs.orphan_pressure, BIN_DEFINITIONS["orphan_pressure"]
            ),
            CrispQLambdaAgent.discretize(
                obs.nearest_orphan_dist, BIN_DEFINITIONS["nearest_orphan_dist"]
            ),
            CrispQLambdaAgent.discretize(
                obs.nearest_orphan_rel_dist, BIN_DEFINITIONS["nearest_orphan_rel_dist"]
            ),
            CrispQLambdaAgent.discretize(obs.truck_load, BIN_DEFINITIONS["truck_load"]),
            CrispQLambdaAgent.discretize(
                obs.fleet_availability, BIN_DEFINITIONS["fleet_availability"]
            ),
            CrispQLambdaAgent.discretize(
                obs.insertion_cost, BIN_DEFINITIONS["insertion_cost"]
            ),
        )
