from .graph import VRPNode, VRPGraph
from .snapshot import *
from .truck import Truck


class VRPEnvironment:

    def __init__(self, graph: VRPGraph):
        self.graph = graph
        self.trucks: dict[int, Truck] = {}

    def add_truck(self, truck: Truck):
        assert truck.id not in self.trucks
        self.trucks[truck.id] = truck

    def get_route(self, truck_id: int) -> list[VRPNode]:
        return [self.graph.get_node(node_id) for node_id in self.trucks[truck_id].route]

    def step(self, agent):
        """
        Runs one step of the simulation of the environment
        """
        return

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
                status=TruckStatusSnapshot(truck.status),
                rel_load=truck.load,
            )
            for truck in self.trucks.values()
        ]

        depot_ss = DepotSnapshot(pos=self.graph.depot.pos)

        routes_ss = [
            [depot_ss.pos]
            + [node.pos for node in self.get_route(truck_id)]
            + [depot_ss.pos]
            for truck_id in self.trucks
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
