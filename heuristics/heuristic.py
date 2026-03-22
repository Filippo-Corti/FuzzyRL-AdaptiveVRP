from abc import ABC, abstractmethod


class Heuristic(ABC):

    @staticmethod
    @abstractmethod
    def can_handle(env: "VRPEnvironment", truck: "Truck") -> bool:
        """
        Return True if this heuristic can handle the given environment
        """
        pass

    @staticmethod
    @abstractmethod
    def apply(env: "VRPEnvironment", truck: "Truck") -> None:
        """
        Apply the heuristic to the given environment, modifying it in-place
        """
        pass
