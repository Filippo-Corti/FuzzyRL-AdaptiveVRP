from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import EnvObservation
    from heuristics import HeuristicAction
    from simulation.snapshot import AgentSnapshot


class VRPAgent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_action(
        self, observation: EnvObservation, available_actions: list[HeuristicAction]
    ) -> HeuristicAction:
        """
        Selects an action for the given truck based on the current state of the environment and the available actions.
        """
        pass

    @abstractmethod
    def update(
        self,
        obs: EnvObservation,
        reward: float,
        available_actions: list[HeuristicAction],
    ):
        """
        Call this after the environment has transitioned as a result of the last select_action call.
        """
        pass

    @abstractmethod
    def get_snapshot(self) -> AgentSnapshot:
        """
        Returns a snapshot of the agent's internal state for visualization purposes.
        """
        pass
