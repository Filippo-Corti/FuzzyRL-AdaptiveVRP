import numpy as np
from typing import Optional


class VRPEnvironment:
    """
    Single-truck constructive VRP.

    The agent picks which of the K nearest unvisited nodes to visit,
    or returns to depot. This gives it real geometric decisions.

    Actions  0 .. K-1 : visit the i-th nearest unvisited node
    Action   K        : RETURN to depot

    Reward: -total_route_distance at episode end (Monte-Carlo).
    """

    def __init__(
        self, n_nodes=15, capacity=10, k_nearest=3, seed: Optional[int] = None
    ):
        self.n_nodes = n_nodes
        self.capacity = capacity
        self.k_nearest = k_nearest
        self.N_ACTIONS = k_nearest + 1
        self.ACTION_RETURN = k_nearest
        self.rng = np.random.default_rng(seed)
        self.depot = np.array([0.5, 0.5])

        # initialised in reset()
        self.node_positions = self.node_demands = None
        self.current_pos = self.visited = None
        self.remaining_capacity = self.current_heading = None
        self.total_distance = self.steps_in_subroute = 0
        self.done = False

    # ------------------------------------------------------------------ API

    def reset(self, node_positions=None, node_demands=None):
        if node_positions is not None:
            self.node_positions = np.array(node_positions, dtype=float)
            self.node_demands = np.array(node_demands, dtype=int)
        else:
            self.node_positions = self.rng.uniform(0, 1, (self.n_nodes, 2))
            self.node_demands = self.rng.integers(1, 6, self.n_nodes)

        self.current_pos = self.depot.copy()
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.remaining_capacity = self.capacity
        self.current_heading = None
        self.total_distance = 0.0
        self.steps_in_subroute = 0
        self.done = False
        return self._get_obs()

    def step(self, action: int):
        assert not self.done
        obs = self._get_obs()
        candidates = obs["candidates"]

        # Force return when nothing feasible remains
        if len(candidates) == 0 or self.remaining_capacity == 0:
            action = self.ACTION_RETURN

        # Clamp to available candidates
        if action < self.ACTION_RETURN:
            action = min(action, len(candidates) - 1)

        if action == self.ACTION_RETURN:
            dist = _dist(self.current_pos, self.depot)
            self.total_distance += dist
            self.current_pos = self.depot.copy()
            self.remaining_capacity = self.capacity
            self.current_heading = None
            self.steps_in_subroute = 0
            if np.all(self.visited):
                self.done = True
        else:
            gidx, _ = candidates[action]
            if self.node_demands[gidx] > self.remaining_capacity:
                return self.step(self.ACTION_RETURN)

            tgt = self.node_positions[gidx]
            dist = _dist(self.current_pos, tgt)
            self.total_distance += dist
            if dist > 1e-9:
                self.current_heading = (tgt - self.current_pos) / dist
            self.current_pos = tgt.copy()
            self.visited[gidx] = True
            self.remaining_capacity -= self.node_demands[gidx]
            self.steps_in_subroute += 1

            if np.all(self.visited):
                close = _dist(self.current_pos, self.depot)
                self.total_distance += close
                self.current_pos = self.depot.copy()
                self.done = True

        reward = -self.total_distance if self.done else 0.0
        next_obs = self._get_obs()
        return next_obs, reward, self.done, {"total_distance": self.total_distance}

    # --------------------------------------------------------------- Features

    def _get_obs(self):
        unvisited = np.where(~self.visited)[0]

        if len(unvisited) == 0:
            return dict(
                remaining_capacity_frac=self.remaining_capacity / self.capacity,
                nearest_dist=0.0,
                angle_to_nearest=0.0,
                second_vs_first_dist=1.0,
                candidates=[],
            )

        dists = np.linalg.norm(
            self.node_positions[unvisited] - self.current_pos, axis=1
        )
        order = np.argsort(dists)
        k = min(self.k_nearest, len(unvisited))
        candidates = [
            (int(unvisited[order[i]]), float(dists[order[i]])) for i in range(k)
        ]

        nd = candidates[0][1]
        if self.current_heading is not None and nd > 1e-9:
            v = (self.node_positions[candidates[0][0]] - self.current_pos) / nd
            cos_ang = float(np.dot(self.current_heading, v))
        else:
            cos_ang = 0.0

        spread = (
            min(candidates[1][1] / max(nd, 1e-9), 5.0) if len(candidates) >= 2 else 1.0
        )

        return dict(
            remaining_capacity_frac=self.remaining_capacity / self.capacity,
            nearest_dist=nd,
            angle_to_nearest=cos_ang,
            second_vs_first_dist=spread,
            candidates=candidates,
        )

    def render_state(self):
        return dict(
            node_positions=self.node_positions.copy(),
            node_demands=self.node_demands.copy(),
            depot=self.depot.copy(),
            current_pos=self.current_pos.copy(),
            visited=self.visited.copy(),
            remaining_capacity=self.remaining_capacity,
            total_distance=self.total_distance,
            done=self.done,
        )


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
