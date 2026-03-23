from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import VRPEnvironment, Truck, EnvObservation

import random


class VRPAgent:

    def __init__(self):
        pass

    def select_action(
        self, observation: EnvObservation, available_actions: list[str]
    ) -> str:
        """
        Selects an action for the given truck based on the current state of the environment and the available actions.
        """
        return random.choice(available_actions)

    def update(self, obs: EnvObservation, reward: float, available_actions: list[str]):
        """
        Call this after the environment has transitioned as a result of the last select_action call.
        """
        pass

    def get_stats(self) -> dict[str, float]:
        """
        Return a dictionary of stats to be logged and plotted during training. This can include things like epsilon, q-table size, etc.
        """
        return {}
