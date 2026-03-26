from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import EnvObservation
    from simulation.snapshot import AgentSnapshot


class VRPAgent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_node(self, obs: EnvObservation) -> int:
        """
        Selects the id of the next node to visit based on the current observation of the environment.
        If -1, the agent chooses to return to the depot.
        """
        pass

    @abstractmethod
    def update(self, reward: float, obs: EnvObservation):
        """
        Receive reward based on the action taken
        """
        pass

    @abstractmethod
    def finish_episode(self):
        """
        Called at the end of an episode, can be used for cleanup or learning updates.
        """
        pass

    @abstractmethod
    def snapshot(self) -> AgentSnapshot:
        """
        Returns a snapshot of the agent's internal state for visualization purposes.
        """
        pass
