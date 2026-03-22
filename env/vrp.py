from .graph import VRPNode, VRPGraph
from .state import *
from .truck import Truck


class VRPEnvironment:

    def __init__(self, graph: VRPGraph):
        self.graph = graph
        self.trucks = []

    def add_truck(self, truck: Truck):
        self.trucks.append(truck)

    def step(self, agent):
        """
        Runs one step of the simulation of the environment
        """
        return

    def get_render_state(self) -> SimulationState:
        """
        Returns the current state of the environment
        :return: the state of the environment
        """

        return SimulationState(
            nodes=[
                NodeState(id=0, pos=(0.2, 0.3), status=NodeStatusState.ASSIGNED),
                NodeState(id=1, pos=(0.5, 0.6), status=NodeStatusState.ASSIGNED),
                NodeState(id=2, pos=(0.7, 0.2), status=NodeStatusState.ASSIGNED),
                NodeState(id=3, pos=(0.4, 0.8), status=NodeStatusState.ASSIGNED),
                NodeState(id=4, pos=(0.8, 0.7), status=NodeStatusState.ASSIGNED),
                NodeState(id=5, pos=(0.3, 0.7), status=NodeStatusState.ASSIGNED),
                NodeState(id=6, pos=(0.2, 0.6), status=NodeStatusState.ASSIGNED),
                NodeState(id=7, pos=(0.7, 0.6), status=NodeStatusState.ASSIGNED),
                NodeState(id=8, pos=(0.5, 0.9), status=NodeStatusState.ASSIGNED),
                NodeState(id=9, pos=(0.1, 0.8), status=NodeStatusState.ASSIGNED),
                NodeState(id=10, pos=(0.3, 0.3), status=NodeStatusState.ASSIGNED),
                NodeState(id=11, pos=(0.1, 0.7), status=NodeStatusState.ASSIGNED),
                NodeState(id=12, pos=(0.7, 0.1), status=NodeStatusState.ASSIGNED),
                NodeState(id=13, pos=(0.4, 0.8), status=NodeStatusState.ASSIGNED),
            ],
            edges=[],  # Fully connected
            trucks=[
                TruckState(
                    id=0, pos=(0.2, 0.3), status=TruckStatusState.ACTIVE, rel_load=0.7
                ),
                TruckState(
                    id=1, pos=(0.5, 0.6), status=TruckStatusState.ACTIVE, rel_load=0.5
                ),
                TruckState(
                    id=2,
                    pos=(0.7, 0.2),
                    status=TruckStatusState.ACTIVE,
                    rel_load=0.2,
                ),
            ],
            routes=[
                [
                    (0.5, 0.5),
                    (0.3, 0.3),
                    (0.2, 0.3),
                    (0.2, 0.6),
                    (0.1, 0.7),
                    (0.3, 0.7),
                    (0.5, 0.5),
                ],
                [
                    (0.5, 0.5),
                    (0.5, 0.6),
                    (0.4, 0.8),
                    (0.5, 0.9),
                    (0.8, 0.7),
                    (0.5, 0.5),
                ],
                [(0.5, 0.5), (0.7, 0.2), (0.7, 0.1), (0.5, 0.5)],
            ],
            depot=DepotState(pos=(0.5, 0.5)),
            stats=SimulationStats(
                round=3,
                orphans=2,
                total_nodes=5,
                total_trucks=3,
                active_trucks=1,
                total_distance=124.7,
                episode_reward=-42.3,
            ),
            agent_state=AgentState(
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
