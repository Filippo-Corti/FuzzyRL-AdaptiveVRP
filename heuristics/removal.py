from env import VRPEnvironment, Truck
from heuristics.heuristic import Heuristic


class CostliestRemoval(Heuristic):

    @staticmethod
    def can_handle(env: VRPEnvironment, truck: Truck) -> bool:
        if truck.route_size == 0:
            return False
        return True

    @staticmethod
    def apply(env: VRPEnvironment, truck: Truck) -> None:
        """
        Performs removal on the given truck's route.
        The heuristic identifies the node in this truck's current planned route whose removal yields the greatest
        reduction in route distance and drops it, making it an orphan. Appropriate when the truck carries a
        geometrically poor node that would be better served by another truck.
        """
        route = env.get_route(truck.id)

        best_node = None
        best_gain = 0.0

        # Iterate over triplets of consecutive nodes and check the gain from removing the middle one
        for A, B, C in zip(route, route[1:], route[2:]):
            gain = A.distance_to(B) + B.distance_to(C) - A.distance_to(C)
            if gain > best_gain:
                best_gain = gain
                best_node = B

        if best_node is None:
            return

        truck.remove_by_id(best_node.id)
        best_node.assignment = None
