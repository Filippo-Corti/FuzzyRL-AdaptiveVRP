from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

from ..agent.base import BaseAgent
from ..env.batch_env import BatchVRPEnv
from ..simulation.snapshot import (
    AgentSnapshot,
    DepotSnapshot,
    EnvironmentSnapshot,
    NodeSnapshot,
    SimulationSnapshot,
    SimulationStats,
    TruckSnapshot,
)


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
        speed: float = 0.05,
        seed: int = 42,
    ):
        if speed <= 0 or speed > 1:
            raise ValueError("speed must be in (0, 1]")

        self.agent = agent
        self.num_nodes = num_nodes
        self.device = device
        self._speed = speed
        self._seed: int | None = seed

        self.env = BatchVRPEnv(batch_size=1, num_nodes=num_nodes, device=device)

        # Animation state
        self._from_xy: tuple[float, float] = (0.0, 0.0)
        self._to_xy: tuple[float, float] = (0.0, 0.0)
        self._t: float = 1.0
        self._step_speed: float = 1.0

        self._pending_action: torch.Tensor | None = None
        self._route_history: list[tuple[float, float]] = []

        self._total_distance: float = 0.0
        self._step_count: int = 0
        self._done: bool = False
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
        self._current_snapshot = None

        depot = self.env.depot_xy[0]
        start = (depot[0].item(), depot[1].item())
        self._route_history.append(start)

        self._from_xy = start
        self._pending_action = self._select_action()
        self._to_xy = self._action_to_xy(self._pending_action)
        self._t = 0.0
        self._step_speed = self._compute_step_speed()

    def randomize(self) -> None:
        """Reset with a new random seed, producing a fresh instance."""
        self._seed = int(torch.randint(0, 2**31, (1,)).item())
        self.reset()

    def microstep(self) -> SimulationSnapshot:
        """
        Advance animation by one frame and return the current snapshot.
        If the truck arrives at its destination, commits the env step and
        selects the next action before returning.
        """
        if self._done:
            return self._build_snapshot()

        self._t = min(self._t + self._step_speed, 1.0)

        if self._t >= 1.0:
            self._commit_step()

        self._current_snapshot = self._build_snapshot()
        return self._current_snapshot

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
        ...

    @classmethod
    @abstractmethod
    def load_agent(
        cls, checkpoint_path: str, num_nodes: int, **kwargs
    ) -> "BaseVisualization":
        """Construct a Visualization from a saved checkpoint."""
        ...

    # ------------------------------------------------------------------
    # Shared animation internals
    # ------------------------------------------------------------------

    def _action_to_xy(self, action: torch.Tensor) -> tuple[float, float]:
        assert self.env.depot_xy is not None
        assert self.env.node_xy is not None
        a = action[0].item()
        if a == 0:
            xy = self.env.depot_xy[0]
        else:
            xy = self.env.node_xy[0, int(a) - 1]
        return (xy[0].item(), xy[1].item())

    def _commit_step(self) -> None:
        assert self._pending_action is not None
        reward = self.env.step(self._pending_action)
        self._total_distance += -reward[0].item()
        self._step_count += 1

        arrival = self._action_to_xy(self._pending_action)
        self._route_history.append(arrival)
        self._from_xy = arrival

        if self.env.all_done():
            self._done = True
            self._t = 1.0
            return

        self._pending_action = self._select_action()
        self._to_xy = self._action_to_xy(self._pending_action)
        self._t = 0.0
        self._step_speed = self._compute_step_speed()

    def _compute_step_speed(self) -> float:
        dx = self._to_xy[0] - self._from_xy[0]
        dy = self._to_xy[1] - self._from_xy[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return 1.0
        return min(self._speed / dist, 1.0)

    def _interpolated_truck_xy(self) -> tuple[float, float]:
        t = self._t
        x = self._from_xy[0] + t * (self._to_xy[0] - self._from_xy[0])
        y = self._from_xy[1] + t * (self._to_xy[1] - self._from_xy[1])
        return (x, y)

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
            load=int((env.capacity[0] - env.remaining_cap[0]).item()),
            capacity=int(env.capacity[0].item()),
            routes=[list(self._route_history)],
        )

        return SimulationSnapshot(
            environment=EnvironmentSnapshot(graph=graph, truck=truck, depot=depot),
            agent=AgentSnapshot(
                last_choice=self._to_xy if not self._done else self._from_xy,
                epsilon=None,
            ),
            stats=SimulationStats(
                round=self._step_count,
                orphans=0,
                total_nodes=self.num_nodes,
                total_distance=self._total_distance,
            ),
        )
