from env import VRPEnvironment, Truck
from .heuristic import Heuristic, HeuristicAction


class DoNothing(Heuristic):

    def is_applicable(self, env: VRPEnvironment, truck: Truck) -> bool:
        return True

    def apply(self, env: VRPEnvironment, truck: Truck) -> None:
        """
        Performs nothing on the given truck's route. This heuristic is a no-op.
        """
        return

    def name(self) -> HeuristicAction:
        return HeuristicAction.DO_NOTHING
