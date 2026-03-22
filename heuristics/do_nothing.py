from env import VRPEnvironment, Truck
from heuristics.heuristic import Heuristic


class DoNothing(Heuristic):

    @staticmethod
    def can_handle(env: VRPEnvironment, truck: Truck) -> bool:
        return True

    @staticmethod
    def apply(env: VRPEnvironment, truck: Truck) -> None:
        """
        Performs nothing on the given truck's route. This heuristic is a no-op.
        """
        return
