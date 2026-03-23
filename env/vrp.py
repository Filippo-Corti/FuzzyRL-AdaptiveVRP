import math
import random
from itertools import pairwise

import config
from agent.agent import VRPAgent
from heuristics.heuristic import Heuristic
from .observation import EnvObservation
from .graph import VRPNode, VRPGraph
from .snapshot import *
from .truck import Truck, TruckStatus


class VRPEnvironment:

    def __init__(self, graph: VRPGraph, heuristics: list[type[Heuristic]]):
        self.graph = graph
        self.trucks: dict[int, Truck] = {}
        self.truck_ids: list[int] = []
        self.current_truck_idx = 0
        self.last_action: str = ""
        self.best_snapshot: SimulationSnapshot | None = None
        self.lowest_cost: float = math.inf
        self.last_cost: float = math.inf
        self.steps = 0
        self.actions: dict[str, type[Heuristic]] = {h.name(): h for h in heuristics}

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

    def step(self, agent: VRPAgent):
        """
        Runs one step of the simulation of the environment
        """
        if len(self.trucks) == 0:
            return

        # Pick and execute action for the current truck
        truck_id = self.truck_ids[self.current_truck_idx]
        truck = self.trucks[truck_id]

        if truck.status == TruckStatus.BROKEN:
            # If the truck is broken, we skip its turn and try to recover it with some probability
            if (
                random.random() < config.RECOVERY_PROB
            ):  # 30% chance to recover each turn
                truck.recover()
            self.current_truck_idx = (self.current_truck_idx + 1) % len(self.truck_ids)
            self.steps += 1
            return
        else:
            # With a certain probability, the truck might break down
            if random.random() < config.DISRUPTION_PROB:
                self.breakdown(truck_id)
                self.current_truck_idx = (self.current_truck_idx + 1) % len(
                    self.truck_ids
                )
                self.steps += 1
                return

        available_actions = [
            k for k, v in self.actions.items() if v.is_applicable(self, truck)
        ]

        action = agent.select_action(self.get_observation(truck), available_actions)
        self.last_action = action
        self.actions[action].apply(self, truck)

        # Compute reward
        reward = self.compute_reward()

        # Advance to next truck before update
        self.current_truck_idx = (self.current_truck_idx + 1) % len(self.truck_ids)
        next_truck = self.trucks[self.truck_ids[self.current_truck_idx]]
        next_available_actions = [
            k for k, v in self.actions.items() if v.is_applicable(self, next_truck)
        ]
        agent.update(self.get_observation(next_truck), reward, next_available_actions)

        # Update counters and stats
        unassigned_nodes = list(self.graph.unassigned_nodes())
        if not unassigned_nodes:
            cost = self.compute_total_distance()
            if cost < self.lowest_cost:
                self.lowest_cost = cost
                self.best_snapshot = self.get_render_state()
            self.last_cost = cost
        self.steps += 1

    def compute_total_distance(self) -> float:
        """
        Computes the total distance of the current planned routes of all trucks
        """
        total_distance = 0.0
        for truck_id in self.truck_ids:
            route = self.get_route(truck_id)
            total_distance += sum(
                A.distance_to(B) for A, B in pairwise(route)
            )  # Sum of distances between consecutive nodes in the route
        return total_distance

    def breakdown(self, truck_id: int):
        """
        Breaks down the truck with the given id, making it unavailable for a certain number of steps and unassigning its nodes
        """
        truck = self.trucks[truck_id]
        for node_id in truck.route:
            self.graph.get_node(node_id).assignment = None
        truck.breakdown()

    def compute_reward(self) -> float:
        """
        Computes the reward for the current state of the environment, based on the current planned routes and the number of unassigned nodes.

        Reward function is:
        R = -total_distance - λ · unvisited_nodes - μ · imbalance$
        """
        total_distance = self.compute_total_distance()
        unvisited_nodes = list(self.graph.unassigned_nodes())
        imbalance = 0.0

        # The average distance between two nodes in the unit square is approximately 0.52, so we can use this as a normalisation factor for the imbalance
        lambda_weight = 10.0

        # Idk why this
        mu_weight = total_distance / 4

        return (
            -total_distance
            - lambda_weight * len(unvisited_nodes)
            - mu_weight * imbalance
        )

    def get_observation(self, truck: Truck) -> EnvObservation:
        """
        Returns the current observation of the environment for the agent,
        from the perspective of the given truck.
        :return: the current observation of the environment
        """
        available_trucks = sum(
            1 for t in self.trucks.values() if t.status != TruckStatus.BROKEN
        )

        orphans = list(self.graph.unassigned_nodes())

        nearest_node = (
            min(  # Find the argmin
                orphans,  # Among all unassigned nodes
                key=lambda unassigned: min(
                    node.distance_to(unassigned) for node in self.get_route(truck.id)
                ),  # Based on the min distance to any other in the route
            )
            if orphans
            else None
        )

        return EnvObservation(
            truck_load=truck.load,
            fleet_availability=(
                available_trucks / len(self.trucks) if self.trucks else 0.0
            ),
            orphan_pressure=(
                len(orphans) / len(self.graph.nodes) if self.graph.nodes else 0.0
            ),
            nearest_orphan_dist=(
                nearest_node.distance_to(nearest_node) / math.sqrt(2)
                if nearest_node
                else 0.0
            ),  # sqrt(2) is the max distance in the unit square
        )

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
                round=self.steps,
                orphans=len(list(self.graph.unassigned_nodes())),
                total_nodes=len(self.graph.nodes),
                total_trucks=len(self.trucks),
                active_trucks=sum(
                    1 for t in self.trucks.values() if t.status != TruckStatus.BROKEN
                ),
                total_distance=self.compute_total_distance(),
                episode_reward=self.compute_reward(),
                last_action=self.last_action,
                truck_turn=self.current_truck_idx,
                best_solution_distance=self.lowest_cost,
                last_distance=self.last_cost,
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
                chosen_action=self.last_action,
                truck_id=0,
            ),
        )
