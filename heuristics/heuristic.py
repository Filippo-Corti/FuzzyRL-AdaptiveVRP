from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env import VRPEnvironment, Truck

from abc import ABC, abstractmethod


class Heuristic(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def is_applicable(self, env: VRPEnvironment, truck: Truck) -> bool:
        """
        Return True if this heuristic can handle the given environment
        """
        pass

    @abstractmethod
    def apply(self, env: VRPEnvironment, truck: Truck) -> None:
        """
        Apply the heuristic to the given environment, modifying it in-place
        """
        pass

    @abstractmethod
    def name(self) -> HeuristicAction:
        """
        Return the name of the heuristic
        """
        pass


class HeuristicAction(StrEnum):
    DO_NOTHING = "Do Nothing"
    NEAREST_INSERTION = "Nearest Insertion"
    COSTLIEST_REMOVAL = "Costliest Removal"
    TWO_OPT = "2-opt"
    CROSS_INSERT = "Cross Insert"
