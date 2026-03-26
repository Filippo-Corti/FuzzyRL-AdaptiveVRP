from simulation.snapshot import *
from .observation import EnvObservation, NodeObservation
from .graph import VRPGraph, VRPNode
from .truck import Truck


class VRPEnvironment:

    def __init__(self, graph: VRPGraph, truck: Truck):
        self.graph = graph
        self.truck = truck

    def get_node_by_id(self, node_id: int) -> VRPNode:
        return self.graph.get_node(node_id)

    def back_to_depot(self):
        self.truck.back_to_depot()
        self.truck.pos = self.graph.depot.pos

    def visit_node(self, node: VRPNode):
        self.truck.add(node)
        node.visited = True
        self.truck.pos = node.pos

    def compute_total_distance(self):
        """
        Computes the total distance traveled by the truck so far, including returning to the depot.
        """
        routes_edges = self.build_routes_edges()
        total_distance = 0.0
        for route in routes_edges:
            for i in range(len(route) - 1):
                p1 = route[i]
                p2 = route[i + 1]
                total_distance += VRPNode.distance(p1, p2)
        return total_distance

    def build_routes_edges(self) -> list[list[tuple[float, float]]]:
        """
        Converts the truck's routes (list of nodes) to a list of positions (x, y) for visualization
        """
        routes = []
        for route in self.truck.routes:
            # Start from the depot
            positions = [self.graph.depot.pos]
            # Add positions of the nodes in the route
            for node in route:
                positions.append(node.pos)
            # Return to the depot
            if route.is_closed:
                positions.append(self.graph.depot.pos)
            routes.append(positions)
        return routes

    def get_observation(self) -> EnvObservation:
        """
        Returns the current observation of the environment for the agent
        """
        d = self.graph.depot
        nodes_obs = [
            NodeObservation(n.id, n.x, n.y, n.demand, n.visited, depot=False)
            for n in self.graph.nodes.values()
        ] + [NodeObservation(d.id, d.x, d.y, d.demand, d.visited, depot=True)]

        return EnvObservation(
            nodes=nodes_obs,
            truck_pos=self.truck.pos,
            truck_load=self.truck.current_load,
            truck_capacity=self.truck.capacity,
        )

    def snapshot(self) -> EnvironmentSnapshot:
        """
        Returns a snapshot of the environment
        """

        nodes_ss = [
            NodeSnapshot(
                n.id,
                n.pos,
                n.demand,
                n.visited,
            )
            for n in self.graph.nodes.values()
        ]

        truck_ss = TruckSnapshot(
            id=self.truck.id,
            pos=self.truck.pos,
            load=self.truck.current_load,
            capacity=self.truck.capacity,
            routes=self.build_routes_edges(),
        )

        depot_ss = DepotSnapshot(pos=self.graph.depot.pos)

        return EnvironmentSnapshot(
            graph=nodes_ss,
            truck=truck_ss,
            depot=depot_ss,
        )

    def reset(self):
        """
        Resets the environment to the initial state
        """
        self.truck.reset()
        self.truck.pos = self.graph.depot.pos
        self.graph.reset()
