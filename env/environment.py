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

        # Nearest orphan distance
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

        # Nearest orphan relative distance
        # How does this truck compare to other trucks in terms of proximity to the nearest orphan?
        # < 1.0 means this truck is closer than average, > 1.0 means it is further
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
                nearest_orphan_rel_dist = 1.0  # only truck, no comparison possible
        else:
            nearest_orphan_rel_dist = 1.0

        # Route efficiency
        # Ratio of actual route distance to number of nodes in the route
        # Lower means more efficient (shorter distance per node)
        # Normalised by the average edge distance in the unit square (~0.52)
        if truck.route_size > 0:
            route_distance = sum(
                route[i].distance_to(route[i + 1]) for i in range(len(route) - 1)
            )
            avg_distance_per_node = route_distance / truck.route_size
            route_efficiency = avg_distance_per_node / 0.52  # normalise by expected avg
        else:
            route_efficiency = 0.0

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
