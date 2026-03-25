import math
from itertools import pairwise

from simulation.snapshot import *
from .observation import EnvObservation
from .graph import VRPNode, VRPGraph
from .truck import Truck, TruckStatus


class VRPEnvironment:

    def __init__(self, graph: VRPGraph, trucks: list[Truck]):
        self.graph = graph
        self.trucks: dict[int, Truck] = {t.id: t for t in trucks}
        self.truck_ids: list[int] = list(self.trucks.keys())
        self.current_truck_idx = 0

    def get_truck_by_index(self, idx: int) -> Truck:
        """Returns the truck corresponding to the given index in the turn order"""
        return self.trucks[self.truck_ids[idx]]

    def get_truck_route(self, truck_id: int) -> list[VRPNode]:
        """Returns full route for the given truck, including the depot as starting and ending point"""
        return (
            [self.graph.depot]
            + [self.graph.get_node(node_id) for node_id in self.trucks[truck_id].route]
            + [self.graph.depot]
        )

    def compute_truck_distance(self, truck_id: int) -> float:
        """Computes the total distance of the route for the given truck"""
        route = self.get_truck_route(truck_id)
        return sum(A.distance_to(B) for A, B in pairwise(route))

    def compute_total_distance(self) -> float:
        """
        Computes the total distance of the current planned routes of all trucks
        """
        return sum(self.compute_truck_distance(truck_id) for truck_id in self.trucks)

    def count_crossings(self) -> int:
        """
        Counts the number of edge crossings between all pairs of truck routes.
        Two segments (A,B) and (C,D) cross if they properly intersect.
        """
        segments = []
        for truck_id in self.trucks:
            route = self.get_truck_route(truck_id)
            for i in range(len(route) - 1):
                segments.append((route[i], route[i + 1]))

        crossings = 0
        for i, (a, b) in enumerate(segments):
            for j, (c, d) in enumerate(segments):
                if j <= i:
                    continue
                if VRPNode.segments_cross(a, b, c, d):
                    crossings += 1
        return crossings

    def compute_imbalance(self) -> float:
        """
        Computes the imbalance of the current load distribution across active trucks, measured as the standard deviation of the loads.
        """
        active_trucks = [
            t for t in self.trucks.values() if t.status == TruckStatus.ACTIVE
        ]
        if len(active_trucks) < 2:
            return 0.0
        loads = [t.load for t in active_trucks]
        mean = sum(loads) / len(loads)
        variance = sum((l - mean) ** 2 for l in loads) / len(loads)
        return variance**0.5  # standard deviation

    def breakdown(self, truck_id: int):
        """
        Breaks down the truck with the given id, making it unavailable for a certain number of steps and unassigning its nodes
        """
        truck = self.trucks[truck_id]
        for node_id in truck.route:
            self.graph.get_node(node_id).assignment = None
        truck.breakdown()

    def recover(self, truck_id: int):
        """
        Recovers the truck with the given id, making it available again
        """
        truck = self.trucks[truck_id]
        truck.recover()

    def get_observation(self, truck: Truck) -> EnvObservation:
        """
        Returns the current observation of the environment for the agent,
        from the perspective of the given truck.
        """

        available_trucks = sum(
            1 for t in self.trucks.values() if t.status != TruckStatus.BROKEN
        )
        orphans = list(self.graph.unassigned_nodes())
        route = self.get_truck_route(truck.id)

        # --- existing fields (unchanged) ---

        if orphans and len(route) > 0:
            nearest_orphan = min(
                orphans,
                key=lambda o: min(node.distance_to(o) for node in route),
            )
            nearest_orphan_dist = min(
                node.distance_to(nearest_orphan) for node in route
            )
        else:
            nearest_orphan = None
            nearest_orphan_dist = 0.0

        if nearest_orphan is not None:
            other_trucks = [
                t
                for t in self.trucks.values()
                if t.id != truck.id and t.status != TruckStatus.BROKEN
            ]
            if other_trucks:
                other_distances = [
                    min(
                        node.distance_to(nearest_orphan)
                        for node in self.get_truck_route(t.id)
                    )
                    for t in other_trucks
                ]
                fleet_avg = sum(other_distances) / len(other_distances)
                nearest_orphan_rel_dist = nearest_orphan_dist / (fleet_avg + 1e-9)
            else:
                nearest_orphan_rel_dist = 1.0
        else:
            nearest_orphan_rel_dist = 1.0

        if truck.route_size > 0:
            route_distance = sum(
                route[i].distance_to(route[i + 1]) for i in range(len(route) - 1)
            )
            avg_distance_per_node = route_distance / truck.route_size
            route_efficiency = avg_distance_per_node / 0.52
        else:
            route_efficiency = 0.0

        # --- new: removal gain ---
        # Best distance saved by removing one node from this truck's route
        # Normalised by average edge distance in unit square
        removal_gain = 0.0
        if len(truck.route) >= 1:
            best_gain = 0.0
            for i, node_id in enumerate(truck.route):
                node = self.graph.get_node(node_id)
                prev_node = route[i]  # route includes depot at index 0
                next_node = route[i + 2]  # route includes depot, so offset by 1
                cost_without = prev_node.distance_to(next_node)
                cost_with = prev_node.distance_to(node) + node.distance_to(next_node)
                gain = cost_with - cost_without
                if gain > best_gain:
                    best_gain = gain
            removal_gain = (
                min(best_gain / 0.52, 2.0) / 2.0
            )  # normalise same as efficiency

        # --- new: route imbalance (fleet-level, same value for all trucks) ---
        route_imbalance = self.compute_imbalance()
        # Normalise: imbalance is std dev of load fractions, max plausible ~0.5
        route_imbalance = min(route_imbalance / 0.5, 1.0)

        # Cheapest insertion cost for the nearest orphan into this truck's route
        insertion_cost = 0.0
        if orphans and len(route) >= 2:
            nearest = min(
                orphans,
                key=lambda o: min(node.distance_to(o) for node in route),
            )
            best_cost = math.inf
            for i in range(len(route) - 1):
                cost = (
                    route[i].distance_to(nearest)
                    + nearest.distance_to(route[i + 1])
                    - route[i].distance_to(route[i + 1])
                )
                if cost < best_cost:
                    best_cost = cost
            insertion_cost = min(best_cost / 0.52, 2.0) / 2.0

        return EnvObservation(
            truck_load=truck.load,
            fleet_availability=(
                available_trucks / len(self.trucks) if self.trucks else 0.0
            ),
            orphan_pressure=(
                len(orphans) / len(self.graph.nodes) if self.graph.nodes else 0.0
            ),
            nearest_orphan_dist=min(nearest_orphan_dist / math.sqrt(2), 1.0),
            nearest_orphan_rel_dist=min(nearest_orphan_rel_dist, 2.0) / 2.0,
            route_efficiency=min(route_efficiency, 2.0) / 2.0,
            removal_gain=removal_gain,
            route_imbalance=route_imbalance,
            insertion_cost=insertion_cost,
        )

    def get_snapshot(self) -> EnvironmentSnapshot:
        """
        Returns a snapshot of the environment
        """

        nodes_ss = [
            NodeSnapshot(
                node.id,
                node.pos,
                (
                    NodeStatusSnapshot.ASSIGNED
                    if node.is_assigned
                    else NodeStatusSnapshot.UNVISITED
                ),
            )
            for node in self.graph
        ]

        trucks_ss = [
            TruckSnapshot(
                id=truck.id,
                pos=truck.pos,
                status=TruckStatusSnapshot(truck.status.value),
                rel_load=truck.load,
            )
            for truck in self.trucks.values()
        ]

        depot_ss = DepotSnapshot(pos=self.graph.depot.pos)

        routes_ss = [
            [node.pos for node in self.get_truck_route(truck_id)]
            for truck_id in self.trucks
        ]

        return EnvironmentSnapshot(
            graph=nodes_ss,
            trucks=trucks_ss,
            routes=routes_ss,
            depot=depot_ss,
        )
