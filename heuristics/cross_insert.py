import math

from env import VRPEnvironment, Truck, TruckStatus
from .heuristic import Heuristic, HeuristicAction


class CrossInsert(Heuristic):
    """
    Atomically moves the highest-removal-gain node from this truck's route
    into the cheapest insertion position of the least-loaded other active truck.
    Never creates an orphan.
    """

    def name(self) -> HeuristicAction:
        return HeuristicAction.CROSS_INSERT

    def is_applicable(self, env: VRPEnvironment, truck: Truck) -> bool:
        if len(truck.route) == 0:
            return False
        # Need at least one other active truck with remaining capacity
        candidates = self._candidate_targets(env, truck)
        return len(candidates) > 0

    def apply(self, env: VRPEnvironment, truck: Truck) -> None:
        targets = self._candidate_targets(env, truck)
        if not targets:
            return

        # Find the node in this truck's route with the highest removal gain
        route = env.get_truck_route(truck.id)
        best_node_idx = None
        best_gain = -math.inf

        for i, node_id in enumerate(truck.route):
            node = env.graph.get_node(node_id)
            prev_node = route[i]  # depot at index 0, so route[i] is prev
            next_node = route[
                i + 2
            ]  # route[i+1] is the node itself, route[i+2] is next
            cost_with = prev_node.distance_to(node) + node.distance_to(next_node)
            cost_without = prev_node.distance_to(next_node)
            gain = cost_with - cost_without
            if gain > best_gain:
                best_gain = gain
                best_node_idx = i

        if best_node_idx is None:
            return

        node_id = truck.route[best_node_idx]
        node = env.graph.get_node(node_id)

        # Find cheapest insertion position in the least-loaded target truck
        target = min(targets, key=lambda t: t.load)
        target_route = env.get_truck_route(target.id)
        best_cost = math.inf
        best_insert_pos = 0

        for i in range(len(target_route) - 1):
            cost = (
                target_route[i].distance_to(node)
                + node.distance_to(target_route[i + 1])
                - target_route[i].distance_to(target_route[i + 1])
            )
            if cost < best_cost:
                best_cost = cost
                best_insert_pos = i

        # Atomically remove from source, insert into target
        truck.route.pop(best_node_idx)
        node.assignment = target.id
        target.route.insert(best_insert_pos, node_id)

    def _candidate_targets(self, env: VRPEnvironment, truck: Truck) -> list[Truck]:
        return [
            t
            for t in env.trucks.values()
            if t.id != truck.id
            and t.status == TruckStatus.ACTIVE
            and t.load < 1.0  # has remaining capacity
        ]
