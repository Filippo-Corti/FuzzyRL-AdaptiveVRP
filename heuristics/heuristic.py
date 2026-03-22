from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import VRPEnvironment, Truck

from abc import ABC, abstractmethod


class Heuristic(ABC):

    @staticmethod
    @abstractmethod
    def is_applicable(env: VRPEnvironment, truck: Truck) -> bool:
        """
        Return True if this heuristic can handle the given environment
        """
        pass

    @staticmethod
    @abstractmethod
    def apply(env: VRPEnvironment, truck: Truck) -> None:
        """
        Apply the heuristic to the given environment, modifying it in-place
        """
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        """
        Return the name of the heuristic
        """
        pass
