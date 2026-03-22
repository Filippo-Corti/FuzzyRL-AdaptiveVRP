from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import VRPEnvironment, Truck

import random


class VRPAgent:

    def __init__(self):
        pass

    def select_action(
        self, env: VRPEnvironment, truck: Truck, available_actions: list[str]
    ) -> str:
        """
        Selects an action for the given truck based on the current state of the environment and the available actions.
        """
        if "Nearest Insertion" in available_actions:
            return "Nearest Insertion"

        # For now, we just select a random action from the available actions
        return random.choice(available_actions)
