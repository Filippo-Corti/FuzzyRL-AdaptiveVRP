from itertools import pairwise

from env import VRPEnvironment, Truck
from heuristics.heuristic import Heuristic


class NearestInsertion(Heuristic):

    @staticmethod
    def is_applicable(env: VRPEnvironment, truck: Truck) -> bool:
        if next(env.graph.unassigned_nodes(), None) is None:
            return False
        if truck.is_full:
            return False
        return True

    @staticmethod
    def apply(env: VRPEnvironment, truck: Truck) -> None:
        """
        Performs nearest insertion on the given truck's route.
        The heuristic finds the position in this truck's current planned route where inserting the nearest
        orphaned node adds the least additional distance. Fast and greedy, appropriate under severe disruption
        when coverage matters more than optimality.
        """
        route = env.get_route(truck.id)

        unassigned_nodes = list(env.graph.unassigned_nodes())

        nearest_node = min(  # Find the argmin
            unassigned_nodes,  # Among all unassigned nodes
            key=lambda unassigned: min(
                node.distance_to(unassigned) for node in route
            ),  # Based on the min distance to any other in the route
        )

        nearest_insertion = min(  # Find the argmin
            pairwise(route),  # Among all pairs of nodes in the route
            key=lambda pair: pair[0].distance_to(nearest_node)
            + nearest_node.distance_to(pair[1])
            - pair[0].distance_to(pair[1]),  # Based on the insertion cost
        )

        # Add at the start if it should go right after the depot
        if nearest_insertion[0].id == env.graph.depot.id:
            truck.add_by_index(nearest_node.id, 0)
        else:
            truck.add_after(nearest_node.id, nearest_insertion[0].id)

        nearest_node.assignment = truck.id

    @staticmethod
    def name() -> str:
        return "Nearest Insertion"
