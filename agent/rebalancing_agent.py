from __future__ import annotations
from typing import TYPE_CHECKING

import config
from agent.crisp_q_lambda import CrispQLambdaAgent

if TYPE_CHECKING:
    from env import EnvObservation

State = tuple[int, int, int, int, int]

BIN_DEFINITIONS: dict[str, list[float]] = {
    "route_imbalance": [0.0, 0.2, 0.4, 0.6, 1.0],
    "truck_load": [0.0, 0.25, 0.5, 0.75, 1.0],
    "route_efficiency": [0.0, 0.25, 0.5, 0.75, 1.0],
    "fleet_availability": [0.0, 0.4, 0.6, 0.8, 1.0],
    "removal_gain": [0.0, 0.1, 0.2, 0.4, 1.0],  # shifted left
}


class RebalancingAgent(CrispQLambdaAgent):

    def decay_epsilon(self):
        self.epsilon = max(
            config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY_REBALANCING
        )

    @staticmethod
    def observation_to_state(obs: EnvObservation) -> State:
        return (
            CrispQLambdaAgent.discretize(
                obs.route_imbalance, BIN_DEFINITIONS["route_imbalance"]
            ),
            CrispQLambdaAgent.discretize(obs.truck_load, BIN_DEFINITIONS["truck_load"]),
            CrispQLambdaAgent.discretize(
                obs.route_efficiency, BIN_DEFINITIONS["route_efficiency"]
            ),
            CrispQLambdaAgent.discretize(
                obs.fleet_availability, BIN_DEFINITIONS["fleet_availability"]
            ),
            CrispQLambdaAgent.discretize(
                obs.removal_gain, BIN_DEFINITIONS["removal_gain"]
            ),
        )
