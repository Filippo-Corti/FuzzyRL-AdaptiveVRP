from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, cast

import config
import torch

from ..agent.base import BaseAgent
from ..env.batch_env import BatchVRPEnv
from .snapshot import (
    AgentSnapshot,
    DepotSnapshot,
    EnvironmentSnapshot,
    NodeSnapshot,
    SimulationSnapshot,
    SimulationStats,
    TruckSnapshot,
)


@dataclass
class MotionSegment:
    """Current full-step segment used by microstep interpolation."""

    origin_xy: tuple[float, float]
    destination_xy: tuple[float, float]


class BaseVisualization(ABC):
    """
    All animation logic lives here — interpolation, history, snapshot
    construction. Subclasses only need to implement _select_action() and
    the load_agent() classmethod.

    The truck moves at constant visual speed between nodes. Each microstep()
    call advances the animation by one frame. The env state advances only
    when the truck reaches its destination.
    """

    def __init__(
        self,
        agent: BaseAgent,
        num_nodes: int,
        device: torch.device,
        env: BatchVRPEnv | None = None,
        speed: float = 0.05,
        seed: int | None = None,
    ):
        if speed <= 0 or speed > 1:
            raise ValueError("speed must be in (0, 1]")

        self.agent = agent
        self.num_nodes = num_nodes
        self.device = device
        self._speed = speed
        self._seed: int | None = seed

        self.env = env or BatchVRPEnv(
            batch_size=1,
            num_nodes=num_nodes,
            device=device,
            depot_mode=cast(Literal["center", "random"], config.ENV_DEPOT_MODE),
            node_xy_range=config.ENV_NODE_XY_RANGE,
            demand_range=config.ENV_DEMAND_RANGE,
            capacity_range=config.ENV_CAPACITY_RANGE,
        )

        # Animation state
        self._segment = MotionSegment(origin_xy=(0.0, 0.0), destination_xy=(0.0, 0.0))
        self._t: float = 1.0
        self._step_speed: float = 1.0

        self._pending_action: torch.Tensor | None = None
        self._route_history: list[tuple[float, float]] = []

        self._total_distance: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
        self._exact_cost: float | None = None
        self._current_snapshot: SimulationSnapshot | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to beginning of episode, using stored seed for reproducibility."""
        if self._seed is not None:
            torch.manual_seed(self._seed)
        self.env.reset()

        assert self.env.depot_xy is not None

        self._route_history = []
        self._total_distance = 0.0
        self._step_count = 0
        self._done = False
        self._exact_cost = self._compute_exact_cost_with_ortools(time_limit_s=1)
        self._current_snapshot = None

        depot = self.env.depot_xy[0]
        start = (depot[0].item(), depot[1].item())
        self._route_history.append(start)

        self._pending_action = self._select_action()
        self._segment = MotionSegment(
            origin_xy=start,
            destination_xy=self._target_xy_for_action(self._pending_action),
        )
        self._t = 0.0
        self._step_speed = self._compute_step_speed()

    def randomize(self) -> None:
        """Reset with a new random seed, producing a fresh instance."""
        self._seed = int(torch.randint(0, 2**31, (1,)).item())
        self.reset()

    def microstep(self) -> None:
        """
        Advance animation by one frame.
        If the truck arrives at its destination, commits the env step and
        selects the next action.
        """
        if self._done:
            self._current_snapshot = self._build_snapshot()
            return

        self._t = min(self._t + self._step_speed, 1.0)

        if self._t >= 1.0:
            self._advance_env_step()

        self._current_snapshot = self._build_snapshot()

    def step_once(self) -> None:
        """Advance exactly one full environment step (to the next node)."""
        if self._done:
            self._current_snapshot = self._build_snapshot()
            return

        # Complete the current segment immediately, then commit one env step.
        self._t = 1.0
        self._advance_env_step()
        self._current_snapshot = self._build_snapshot()

    def current_snapshot(self) -> SimulationSnapshot:
        """Return the most recently built snapshot without advancing state."""
        if self._current_snapshot is None:
            self._current_snapshot = self._build_snapshot()
        return self._current_snapshot

    def is_done(self) -> bool:
        return self._done

    def set_speed(self, speed: float) -> None:
        """Update animation speed in a validated way."""
        if speed <= 0 or speed > 1:
            raise ValueError("speed must be in (0, 1]")
        self._speed = speed

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _select_action(self) -> torch.Tensor:
        """
        Query the agent and return a (1,) action tensor.
        Subclasses call self.agent.select_action with whatever preprocessing
        their agent requires.
        """
        pass

    @classmethod
    @abstractmethod
    def load_agent(
        cls, checkpoint_path: str, num_nodes: int, **kwargs
    ) -> "BaseVisualization":
        """Construct a Visualization from a saved checkpoint."""
        pass

    @abstractmethod
    def _reload_agent_state(
        self,
        checkpoint_path: str,
        device: torch.device | None = None,
    ) -> float:
        """Reload model parameters from checkpoint and return a display metric."""
        pass

    def reload_checkpoint(
        self,
        checkpoint_path: str,
        device: torch.device | None = None,
    ) -> float:
        return self._reload_agent_state(checkpoint_path=checkpoint_path, device=device)

    # ------------------------------------------------------------------
    # Shared animation internals
    # ------------------------------------------------------------------

    def _target_xy_for_action(self, action: torch.Tensor) -> tuple[float, float]:
        assert self.env.depot_xy is not None
        assert self.env.node_xy is not None
        a = action[0].item()
        if a == 0:
            xy = self.env.depot_xy[0]
        else:
            xy = self.env.node_xy[0, int(a) - 1]
        return (xy[0].item(), xy[1].item())

    def _advance_env_step(self) -> None:
        assert self._pending_action is not None
        reward = self.env.step(self._pending_action)
        self._total_distance += -reward[0].item()
        self._step_count += 1

        arrival = self._target_xy_for_action(self._pending_action)
        self._route_history.append(arrival)
        self._segment.origin_xy = arrival

        if self.env.all_done():
            self._done = True
            self._t = 1.0
            return

        self._pending_action = self._select_action()
        self._segment.destination_xy = self._target_xy_for_action(self._pending_action)
        self._t = 0.0
        self._step_speed = self._compute_step_speed()

    def _compute_step_speed(self) -> float:
        dx = self._segment.destination_xy[0] - self._segment.origin_xy[0]
        dy = self._segment.destination_xy[1] - self._segment.origin_xy[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return 1.0
        return min(self._speed / dist, 1.0)

    def _interpolated_truck_xy(self) -> tuple[float, float]:
        t = self._t
        x = self._segment.origin_xy[0] + t * (
            self._segment.destination_xy[0] - self._segment.origin_xy[0]
        )
        y = self._segment.origin_xy[1] + t * (
            self._segment.destination_xy[1] - self._segment.origin_xy[1]
        )
        return (x, y)

    def _truck_heading_deg(self) -> float:
        """Heading angle in degrees from the truck's current direction of travel."""
        dx = self._segment.destination_xy[0] - self._segment.origin_xy[0]
        dy = self._segment.destination_xy[1] - self._segment.origin_xy[1]
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            return math.degrees(math.atan2(dy, dx))

        if len(self._route_history) >= 2:
            (x0, y0), (x1, y1) = self._route_history[-2], self._route_history[-1]
            dx2 = x1 - x0
            dy2 = y1 - y0
            if abs(dx2) > 1e-9 or abs(dy2) > 1e-9:
                return math.degrees(math.atan2(dy2, dx2))

        return 0.0

    def _build_snapshot(self) -> SimulationSnapshot:
        truck_xy = self._interpolated_truck_xy()
        env = self.env
        assert env.node_xy is not None
        assert env.node_demands is not None
        assert env.visited is not None
        assert env.depot_xy is not None
        assert env.capacity is not None
        assert env.remaining_cap is not None

        graph = [
            NodeSnapshot(
                id=i + 1,
                pos=(env.node_xy[0, i, 0].item(), env.node_xy[0, i, 1].item()),
                demand=int(env.node_demands[0, i].item()),
                visited=bool(env.visited[0, i].item()),
            )
            for i in range(self.num_nodes)
        ]

        depot_xy = env.depot_xy[0]
        depot = DepotSnapshot(pos=(depot_xy[0].item(), depot_xy[1].item()))

        truck = TruckSnapshot(
            id=0,
            pos=truck_xy,
            heading_deg=self._truck_heading_deg(),
            load=int((env.capacity[0] - env.remaining_cap[0]).item()),
            capacity=int(env.capacity[0].item()),
            routes=[list(self._route_history)],
        )

        return SimulationSnapshot(
            environment=EnvironmentSnapshot(graph=graph, truck=truck, depot=depot),
            agent=AgentSnapshot(
                last_choice=(
                    self._segment.destination_xy
                    if not self._done
                    else self._segment.origin_xy
                ),
                epsilon=None,
            ),
            stats=SimulationStats(
                round=self._step_count,
                orphans=0,
                total_nodes=self.num_nodes,
                total_distance=self._total_distance,
                exact_cost=self._exact_cost,
            ),
        )

    def _compute_exact_cost_with_ortools(
        self,
        time_limit_s: int,
    ) -> float | None:
        """Solve current visualized instance with OR-Tools and return cost."""
        try:
            pywrapcp = import_module("ortools.constraint_solver.pywrapcp")
            routing_enums_pb2 = import_module(
                "ortools.constraint_solver.routing_enums_pb2"
            )
        except Exception:
            return None

        assert self.env.depot_xy is not None
        assert self.env.node_xy is not None
        assert self.env.node_demands is not None
        assert self.env.capacity is not None

        depot = self.env.depot_xy[0]
        customers_xy = self.env.node_xy[0]
        demands = self.env.node_demands[0]
        capacity = int(self.env.capacity[0].item())

        coords: list[tuple[float, float]] = [
            (float(depot[0].item()), float(depot[1].item()))
        ]
        coords.extend(
            (float(customers_xy[i, 0].item()), float(customers_xy[i, 1].item()))
            for i in range(self.num_nodes)
        )
        node_demands = [0]
        node_demands.extend(int(demands[i].item()) for i in range(self.num_nodes))

        # OR-Tools uses integer arc costs; scale Euclidean distances to keep precision.
        dist_scale = 10_000
        n_points = len(coords)
        dist_matrix: list[list[int]] = [[0 for _ in range(n_points)] for _ in range(n_points)]
        for i in range(n_points):
            xi, yi = coords[i]
            for j in range(n_points):
                if i == j:
                    continue
                xj, yj = coords[j]
                d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                dist_matrix[i][j] = int(round(d * dist_scale))

        num_vehicles = self.num_nodes
        manager = pywrapcp.RoutingIndexManager(n_points, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return dist_matrix[from_node][to_node]

        transit_cb_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)

        def demand_callback(from_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            return node_demands[from_node]

        demand_cb_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_cb_index,
            0,
            [capacity] * num_vehicles,
            True,
            "Capacity",
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.FromSeconds(max(1, int(time_limit_s)))

        try:
            solution = routing.SolveWithParameters(search_params)
        except Exception:
            return None

        if solution is None:
            return None

        return float(solution.ObjectiveValue()) / float(dist_scale)
