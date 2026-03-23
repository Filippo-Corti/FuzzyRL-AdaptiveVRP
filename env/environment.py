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

        nearest_node = min(  # Find the argmin
            orphans,  # Among all unassigned nodes
            key=lambda unassigned: min(
                node.distance_to(unassigned) for node in self.get_truck_route(truck.id)
            ),  # Based on the min distance to any other in the route
            default=0.0,
        )

        if nearest_node:
            nearest_orphan_dist = min(
                node.distance_to(nearest_node)
                for node in self.get_truck_route(truck.id)
            )
        else:
            nearest_orphan_dist = 0.0

        return EnvObservation(
            truck_load=truck.load,
            fleet_availability=(
                available_trucks / len(self.trucks) if self.trucks else 0.0
            ),
            orphan_pressure=(
                len(orphans) / len(self.graph.nodes) if self.graph.nodes else 0.0
            ),
            nearest_orphan_dist=(
                nearest_orphan_dist
                / math.sqrt(2)  # sqrt(2) is the max distance in the unit square
            ),
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
