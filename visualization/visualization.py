from __future__ import annotations

import math
import torch
from pathlib import Path
from dataclasses import dataclass, field

from agent.transformer_agent import TransformerAgent
from env.batch_env import BatchVRPEnv
from simulation.snapshot import (
    SimulationSnapshot,
    EnvironmentSnapshot,
    NodeSnapshot,
    TruckSnapshot,
    DepotSnapshot,
    SimulationStats,
    AgentSnapshot,
)


class Visualization:
    """
    Drives a single VRP instance (batch_size=1) through a trained agent,
    producing SimulationSnapshots suitable for the Renderer.

    The truck moves smoothly between nodes. Each call to microstep() advances
    the animation by one frame. The underlying env state advances only when
    the truck reaches its destination (i.e. when the interpolation completes).

    Typical usage from a pygame loop:

        viz = Visualization.load_agent("checkpoints/transformer.pt", num_nodes=15)
        viz.reset()
        while not viz.is_done():
            snapshot = viz.microstep()
            renderer.render(snapshot)
    """

    def __init__(
        self,
        agent: TransformerAgent,
        num_nodes: int,
        device: torch.device,
        speed: float = 0.05,
    ):
        """
        speed: how much t advances per microstep (1/speed = frames per move).
               0.05 → 20 frames per node-to-node movement.
        seed:  RNG seed used by reset(). Same seed → same instance every time.
               Pass seed=None to get a fresh random instance on each reset().
        """
        if speed <= 0 or speed > 1:
            raise ValueError("speed must be in (0, 1]")

        self.agent = agent
        self.num_nodes = num_nodes
        self.device = device
        self._speed = speed

        self.env = BatchVRPEnv(batch_size=1, num_nodes=num_nodes, device=device)

        # Animation state
        self._from_xy: tuple[float, float] = (0.0, 0.0)
        self._to_xy: tuple[float, float] = (0.0, 0.0)
        self._t: float = 1.0  # starts at 1.0 so first microstep triggers an env step

        # Pending action (selected at arrival, executed on next arrival)
        self._pending_action: torch.Tensor | None = None

        # History for TruckSnapshot.routes
        self._route_history: list[tuple[float, float]] = []

        # Running totals for stats
        self._total_distance: float = 0.0
        self._step_count: int = 0

        self._done: bool = False

        # Cache the last built snapshot so current_snapshot() is always valid
        self._current_snapshot: SimulationSnapshot | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Generate a new instance and restart the animation.
        Uses self._seed if set, giving a reproducible instance each call.
        """
        self.env.reset()

        self._route_history = []
        self._total_distance = 0.0
        self._step_count = 0
        self._done = False

        # Truck starts at depot
        depot = self.env.depot_xy[0]
        start = (depot[0].item(), depot[1].item())
        self._route_history.append(start)

        # Select first action immediately so _to_xy is meaningful
        self._from_xy = start
        self._pending_action = self._select_action()
        self._to_xy = self._action_to_xy(self._pending_action)
        self._t = 0.0
        self._step_speed = self._compute_step_speed()

    def microstep(self) -> SimulationSnapshot:
        """
        Advance animation by one frame and return the current snapshot.

        - If the truck is mid-move: advance t, return interpolated position.
        - If t reaches 1.0: commit the pending env step, select the next
          action, begin the next move.
        """
        if self._done:
            return self._build_snapshot()

        self._t = min(self._t + self._step_speed, 1.0)

        if self._t >= 1.0:
            self._commit_step()

        self._current_snapshot = self._build_snapshot()
        return self._current_snapshot

    def randomize(self) -> None:
        """Reset with a fresh random seed (new instance each call)."""
        self._seed = torch.randint(0, 2**31, (1,)).item()
        self.reset()

    def current_snapshot(self) -> SimulationSnapshot:
        """Return the most recently built snapshot without advancing the animation."""
        if self._current_snapshot is None:
            self._current_snapshot = self._build_snapshot()
        return self._current_snapshot

    def is_done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_action(self) -> torch.Tensor:
        """Run the agent on the current env state and return a (1,) action tensor."""
        with torch.no_grad():
            node_features, truck_state, mask = self.env.get_state()
            actions, _ = self.agent.select_action(
                node_features, truck_state, mask, greedy=True
            )
        return actions  # shape (1,)

    def _action_to_xy(self, action: torch.Tensor) -> tuple[float, float]:
        """Convert an action index to world coordinates."""
        a = action[0].item()
        if a == 0:
            xy = self.env.depot_xy[0]
        else:
            xy = self.env.node_xy[0, int(a) - 1]
        return (xy[0].item(), xy[1].item())

    def _commit_step(self) -> None:
        """
        Apply the pending action to the env, update history and totals,
        then select the next action and set up the next move segment.
        """
        # Step env
        reward = self.env.step(self._pending_action)
        dist = -reward[0].item()  # reward is negative distance
        self._total_distance += dist
        self._step_count += 1

        # Record arrival position in history
        arrival = self._action_to_xy(self._pending_action)
        self._route_history.append(arrival)
        self._from_xy = arrival

        if self.env.all_done():
            self._done = True
            self._t = 1.0
            return

        # Select next action and begin new move segment
        self._pending_action = self._select_action()
        self._to_xy = self._action_to_xy(self._pending_action)
        self._t = 0.0
        self._step_speed = self._compute_step_speed()

    def _compute_step_speed(self) -> float:
        """
        Compute t-increment per frame so the truck moves at constant visual speed.
        self._speed is the fraction of the graph diagonal covered per frame.
        A move of distance d takes d/self._speed frames → speed_t = self._speed/d.
        Clamped to [self._speed, 1.0] to avoid zero-length moves stalling.
        """
        dx = self._to_xy[0] - self._from_xy[0]
        dy = self._to_xy[1] - self._from_xy[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return 1.0  # zero-length move, complete instantly
        return min(self._speed / dist, 1.0)

    def _interpolated_truck_xy(self) -> tuple[float, float]:
        t = self._t
        x = self._from_xy[0] + t * (self._to_xy[0] - self._from_xy[0])
        y = self._from_xy[1] + t * (self._to_xy[1] - self._from_xy[1])
        return (x, y)

    def _build_snapshot(self) -> SimulationSnapshot:
        truck_xy = self._interpolated_truck_xy()
        env = self.env

        # Node snapshots (indices 0..N-1 in node_xy)
        graph: list[NodeSnapshot] = []
        for i in range(self.num_nodes):
            xy = env.node_xy[0, i]
            graph.append(
                NodeSnapshot(
                    id=i + 1,
                    pos=(xy[0].item(), xy[1].item()),
                    demand=int(env.node_demands[0, i].item()),
                    visited=env.visited[0, i].item(),
                )
            )

        depot_xy = env.depot_xy[0]
        depot = DepotSnapshot(pos=(depot_xy[0].item(), depot_xy[1].item()))

        truck = TruckSnapshot(
            id=0,
            pos=truck_xy,
            load=int((env.capacity[0] - env.remaining_cap[0]).item()),
            capacity=int(env.capacity[0].item()),
            routes=[list(self._route_history)],  # copy of history so far
        )

        environment = EnvironmentSnapshot(graph=graph, truck=truck, depot=depot)

        agent_snap = AgentSnapshot(
            last_choice=self._to_xy if not self._done else self._from_xy,
            epsilon=None,
        )

        stats = SimulationStats(
            round=self._step_count,
            orphans=0,
            total_nodes=self.num_nodes,
            total_distance=self._total_distance,
        )

        return SimulationSnapshot(
            environment=environment,
            agent=agent_snap,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load_agent(
        cls,
        checkpoint_path: str,
        num_nodes: int,
        d_model: int = 128,
        speed: float = 0.05,
        device: torch.device | None = None,
    ) -> "Visualization":
        """Load a trained agent from a checkpoint and return a ready Visualization."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = TransformerAgent(
            node_features=4, state_features=3, d_model=d_model, device=device
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.eval()

        return cls(agent=agent, num_nodes=num_nodes, device=device, speed=speed)
