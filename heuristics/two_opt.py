from itertools import pairwise, combinations

from env import VRPEnvironment, Truck
from .heuristic import Heuristic, HeuristicAction


class TwoOpt(Heuristic):

    def is_applicable(self, env: VRPEnvironment, truck: Truck) -> bool:
        if truck.route_size < 3:
            return False
        return True

    def apply(self, env: VRPEnvironment, truck: Truck) -> None:
        """
        Performs 2-opt on the given truck's route.
        The heuristic finds the single best improving 2-opt swap across all pairs of edges in this truck's planned
        route and applies it. If no improving swap exists the action has no effect. Reduces route distance without
        changing which nodes belong to this truck. Appropriate when the state is stable and the agent is in
        improvement mode.
        """
        route = env.get_truck_route(truck.id)
        edges = list(pairwise(route))

        best_swap = None
        best_improvement = 0.0
        for (a, b), (c, d) in combinations(edges, r=2):

            # Exclude candidates that involve the depot
            if a.id == env.graph.depot.id or d.id == env.graph.depot.id:
                continue

            # Exclude consecutive edges, as the swap would not change anything
            if b.id == c.id:
                continue

            improvement = (
                a.distance_to(c)
                + b.distance_to(d)
                - a.distance_to(b)
                - c.distance_to(d)
            )

            if improvement < best_improvement:
                best_improvement = improvement
                best_swap = (a, b, c, d)

        if best_swap is None:
            return

        a, b, c, d = best_swap
        truck_route = truck.route.copy()
        b_idx = truck_route.index(b.id)
        c_idx = truck_route.index(c.id)

        # Remove nodes between B and C (inclusive)
        for tid in truck_route[b_idx : c_idx + 1]:
            truck.remove_by_id(tid)

        # Reinsert them between A and D
        for tid in truck_route[b_idx : c_idx + 1]:
            truck.add_after(tid, a.id)

    def name(self) -> HeuristicAction:
        return HeuristicAction.TWO_OPT
