import math
import random
from itertools import pairwise, combinations

from .graph import VRPNode, VRPGraph
from .snapshot import *
from .truck import Truck


class VRPEnvironment:

    def __init__(self, graph: VRPGraph):
        self.graph = graph
        self.trucks: dict[int, Truck] = {}
        self.truck_ids: list[int] = []
        self.current_truck_idx = 0
        self.last_action: str = ""

    def add_truck(self, truck: Truck):
        assert truck.id not in self.trucks
        self.trucks[truck.id] = truck
        self.truck_ids.append(truck.id)

    def get_route(self, truck_id: int) -> list[VRPNode]:
        """Returns full route for the given truck, including the depot as starting and ending point"""
        return (
            [self.graph.depot]
            + [self.graph.get_node(node_id) for node_id in self.trucks[truck_id].route]
            + [self.graph.depot]
        )

    def step(self, agent):
        """
        Runs one step of the simulation of the environment
        """
        if len(self.trucks) == 0:
            return

        truck = self.trucks[self.truck_ids[self.current_truck_idx]]

        # Perform an action each with uniform probability: nearest insertion, 2-opt, do nothing
        p = random.random()
        if p < 4 / 10:
            self.nearest_insertion(truck)
            self.last_action = "Nearest Insertion"
        elif p < 6 / 10:
            self.two_opt(truck)
            self.last_action = "2-opt"
        elif p < 8 / 10:
            self.remove_costliest(truck)
            self.last_action = "Remove Costliest"
        else:
            self.last_action = "Do Nothing"

        self.current_truck_idx = (self.current_truck_idx + 1) % len(self.truck_ids)
        return

    def nearest_insertion(self, truck: Truck):
        """
        Performs nearest insertion on the given truck's route.
        The heuristic finds the position in this truck's current planned route where inserting the nearest
        orphaned node adds the least additional distance. Fast and greedy, appropriate under severe disruption
        when coverage matters more than optimality.
        """
        route = self.get_route(truck.id)

        unassigned_nodes = list(self.graph.unassigned_nodes())
        if truck.is_full or not unassigned_nodes:
            return

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

        # I should be careful with the depot unfortunately
        if nearest_insertion[0].id == self.graph.depot.id:
            truck.add_by_index(nearest_node.id, 0)  # Add at the start
        else:
            truck.add_after(nearest_node.id, nearest_insertion[0].id)

        # Either way, we need to assign the node to the truck
        nearest_node.assignment = truck.id

    def two_opt(self, truck: Truck):
        """
        Performs 2-opt on the given truck's route.
        The heuristic finds the single best improving 2-opt swap across all pairs of edges in this truck's planned
        route and applies it. If no improving swap exists the action has no effect. Reduces route distance without
        changing which nodes belong to this truck. Appropriate when the state is stable and the agent is in
        improvement mode.
        """

        # I should iterate over all pairwise nodes in the route
        # Two times nested
        # And I should check what the improvement for making the swap would be (this is easy cause symmetric)
        # Then, I find the max improvement --> (A, B) and (C. D) should be swapped
        # To adopt the change I should remove all nodes between B and C (inclusive) and reinsert them between A and D
        # I should be careful with the depot unfortunately, but I can just exclude it from the swap candidates

        route = self.get_route(truck.id)
        edges = list(pairwise(route))

        best_swap = None
        best_improvement = 0.0

        for (A, B), (C, D) in combinations(edges, r=2):
            if A.id == self.graph.depot.id or D.id == self.graph.depot.id:
                continue

            # If they are adjacent, the swap would not change anything, so I can skip it
            if B.id == C.id:
                continue

            improvement = (
                A.distance_to(C)
                + B.distance_to(D)
                - A.distance_to(B)
                - C.distance_to(D)
            )

            if improvement < best_improvement:
                best_improvement = improvement
                best_swap = (A, B, C, D)

        if best_swap is None:
            return
        A, B, C, D = best_swap
        # I should remove all nodes between B and C (inclusive) and reinsert them between A and D
        # I can find the indices of B and C in the route and then do the operations on the truck route
        truck_route = truck.route.copy()
        idx_B = truck_route.index(B.id)
        idx_C = truck_route.index(C.id)

        # Remove nodes between B and C (inclusive)
        for id in truck_route[idx_B : idx_C + 1]:
            truck.remove_by_id(id)

        # Reinsert them between A and D
        for id in truck_route[idx_B : idx_C + 1]:
            truck.add_after(id, A.id)

    def remove_costliest(self, truck: Truck):
        """
        Performs removal on the given truck's route.
        The heuristic identifies the node in this truck's current planned route whose removal yields the greatest
        reduction in route distance and drops it, making it an orphan. Appropriate when the truck carries a
        geometrically poor node that would be better served by another truck.
        """
        route = self.get_route(truck.id)

        if truck.route_size == 0:
            return

        best_node = None
        best_gain = 0.0

        for A, B, C in zip(route, route[1:], route[2:]):
            gain = A.distance_to(B) + B.distance_to(C) - A.distance_to(C)
            if gain > best_gain:
                best_gain = gain
                best_node = B

        if best_node is None:
            return

        truck.remove_by_id(best_node.id)
        best_node.assignment = None

    def get_render_state(self) -> SimulationSnapshot:
        """
        Returns a snapshot of the environment
        :return: the snapshot of the environment
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
            [node.pos for node in self.get_route(truck_id)] for truck_id in self.trucks
        ]

        return SimulationSnapshot(
            nodes=nodes_ss,
            trucks=trucks_ss,
            routes=routes_ss,
            depot=depot_ss,
            stats=SimulationStats(
                round=3,
                orphans=2,
                total_nodes=5,
                total_trucks=3,
                active_trucks=1,
                total_distance=124.7,
                episode_reward=-42.3,
                last_action=self.last_action,
                truck_turn=self.current_truck_idx,
            ),
            agent_state=AgentSnapshot(
                memberships={
                    "truck_load": {
                        "Empty": 0.0,
                        "Plenty": 0.6,
                        "Tight": 0.4,
                        "Almost Gone": 0.0,
                    },
                    "fleet_avail": {
                        "Full": 0.0,
                        "Reduced": 0.8,
                        "Critically Reduced": 0.2,
                    },
                    "orphan_pressure": {
                        "None": 0.0,
                        "Low": 0.3,
                        "Moderate": 0.7,
                        "High": 0.0,
                    },
                    "nearest_orphan": {"Near": 0.1, "Medium": 0.9, "Far": 0.0},
                },
                q_values={
                    "do_nothing": -12.4,
                    "insert_nearest_cheapest": -8.1,
                    "insert_nearest_regret": -9.3,
                    "two_opt": -11.0,
                    "swap_overloaded": -10.2,
                },
                chosen_action="insert_nearest_cheapest",
                truck_id=0,
            ),
        )
