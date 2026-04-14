from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..base import AgentDecision, AgentObservation, BaseAgent


# ---------------------------------------------------------------------------
# Fuzzy primitives
# ---------------------------------------------------------------------------

def _triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function: peak at b, zero at a and c."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def fuzzify(value: float, breakpoints: list[tuple[str, float, float, float]]) -> dict[str, float]:
    """Return {label: membership} for all non-zero memberships."""
    result = {}
    for label, a, b, c in breakpoints:
        mu = _triangular(value, a, b, c)
        if mu > 0.0:
            result[label] = mu
    return result


# ---------------------------------------------------------------------------
# Signal definitions
#
# Five geometrically meaningful signals for single-truck static VRP:
#
#   dist_nearest      — normalised distance from truck to nearest unvisited node
#                       low  → greedy move is cheap
#                       high → nearest option is already expensive
#
#   isolation_nearest — normalised distance from nearest unvisited node to its
#                       own nearest unvisited neighbour (excluding itself)
#                       high → that node is a loner; defer and it gets expensive
#
#   detour_nearest    — extra distance of visiting nearest vs heading directly
#                       toward the centroid of all remaining unvisited nodes,
#                       normalised by that centroid distance
#                       high → nearest node pulls away from remaining work
#
#   dist_to_depot     — normalised distance from truck to depot
#                       high → far from natural loop-close point
#
#   unvisited_fraction — fraction of customer nodes not yet visited
#                       high → early tour; low → near the end
# ---------------------------------------------------------------------------

BREAKPOINTS: dict[str, list[tuple[str, float, float, float]]] = {
    "dist_nearest": [
        ("L", 0.0, 0.0, 0.4),
        ("M", 0.2, 0.5, 0.8),
        ("H", 0.6, 1.0, 1.0),
    ],
    "isolation_nearest": [
        ("L", 0.0, 0.0, 0.4),
        ("M", 0.2, 0.5, 0.8),
        ("H", 0.6, 1.0, 1.0),
    ],
    "detour_nearest": [
        ("L", 0.0, 0.0, 0.4),
        ("M", 0.2, 0.5, 0.8),
        ("H", 0.6, 1.0, 1.0),
    ],
    "dist_to_depot": [
        ("L", 0.0, 0.0, 0.4),
        ("M", 0.2, 0.5, 0.8),
        ("H", 0.6, 1.0, 1.0),
    ],
    "unvisited_fraction": [
        ("L", 0.0, 0.0, 0.4),
        ("M", 0.2, 0.5, 0.8),
        ("H", 0.6, 1.0, 1.0),
    ],
}

SIGNAL_NAMES = list(BREAKPOINTS.keys())

# ---------------------------------------------------------------------------
# Action space  (semantic — stable meaning regardless of instance state)
#
#   VISIT_NEAREST    — visit the unvisited node closest to the truck right now
#   VISIT_ISOLATED   — visit the unvisited node farthest from its own neighbours
#                      (highest isolation); prevents expensive end-of-tour detours
#   VISIT_DETOUR     — visit the unvisited node with the lowest detour cost
#                      relative to the centroid of remaining nodes; keeps the
#                      truck heading toward remaining work
#
# The heuristic resolves which specific node index each action maps to at
# runtime. The agent never selects a raw node index.
# ---------------------------------------------------------------------------

ACTION_VISIT_NEAREST  = 0
ACTION_VISIT_ISOLATED = 1
ACTION_VISIT_DETOUR   = 2
N_ACTIONS = 3


class FuzzyAgent(BaseAgent):
    """
    Fuzzy Q-learning agent for single-truck static VRP tour construction.

    State: five geometric signals capturing local cost, isolation pressure,
    detour penalty, depot proximity, and tour progress.

    Actions: three semantic heuristics (nearest, most-isolated, lowest-detour).
    The agent learns when to deviate from greedy nearest-neighbour.
    """

    def __init__(
        self,
        epsilon: float = 0.9,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        lr: float = 0.1,
        gamma: float = 0.95,
    ):
        print(
            f"[fuzzy agent] init epsilon={epsilon}, epsilon_min={epsilon_min}, "
            f"epsilon_decay={epsilon_decay}, lr={lr}, gamma={gamma}"
        )
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.gamma = gamma

        self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * N_ACTIONS)
        self._last_state_weights: list[tuple[tuple, float]] | None = None
        self._last_action: int | None = None

    # ------------------------------------------------------------------
    # Public interface (matches BaseAgent contract)
    # ------------------------------------------------------------------

    def select_action(
        self,
        observation: AgentObservation,
        greedy: bool = False,
    ) -> AgentDecision:
        signals, action_targets = self._extract_signals(
            observation.node_features,
            observation.truck_state,
            observation.mask,
        )

        if signals is None:
            # No unvisited nodes — episode over, return depot (index 0)
            self._last_state_weights = None
            self._last_action = None
            return AgentDecision(actions=torch.tensor([0]), log_probs=torch.zeros(1))

        assert action_targets is not None
        state_weights = self._get_state_weights(signals)
        q_values = self._aggregate_q(state_weights)

        if not greedy and np.random.random() < self.epsilon:
            chosen_action = np.random.randint(N_ACTIONS)
        else:
            chosen_action = int(np.argmax(q_values))

        self._last_state_weights = state_weights
        self._last_action = chosen_action

        node_index = action_targets[chosen_action]
        return AgentDecision(
            actions=torch.tensor([node_index]),
            log_probs=torch.zeros(1),
        )

    def q_update(
        self,
        reward: float,
        next_node_features: torch.Tensor,
        next_truck_state: torch.Tensor,
        next_mask: torch.Tensor,
        done: bool,
    ) -> None:
        if self._last_state_weights is None or self._last_action is None:
            return

        if done:
            target = reward
        else:
            next_signals, _ = self._extract_signals(
                next_node_features, next_truck_state, next_mask
            )
            if next_signals is None:
                target = reward
            else:
                next_weights = self._get_state_weights(next_signals)
                next_q = self._aggregate_q(next_weights)
                target = reward + self.gamma * max(next_q)

        # Normalise weights before updating so that the effective learning rate
        # is independent of how many fuzzy combinations are active.
        total_weight = sum(w for _, w in self._last_state_weights)
        for state_key, weight in self._last_state_weights:
            normalised_w = weight / total_weight
            current_q = self.q_table[state_key][self._last_action]
            self.q_table[state_key][self._last_action] += (
                self.lr * normalised_w * (target - current_q)
            )

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def last_fuzzy_action(self) -> int | None:
        return self._last_action

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": dict(self.q_table),
                    "epsilon": self.epsilon,
                    "epsilon_min": self.epsilon_min,
                    "epsilon_decay": self.epsilon_decay,
                    "lr": self.lr,
                    "gamma": self.gamma,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "FuzzyAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            epsilon=float(data.get("epsilon", 0.9)),
            epsilon_min=float(data.get("epsilon_min", 0.05)),
            epsilon_decay=float(data.get("epsilon_decay", 0.9995)),
            lr=float(data.get("lr", 0.1)),
            gamma=float(data.get("gamma", 0.95)),
        )
        agent.q_table = defaultdict(lambda: [0.0] * N_ACTIONS, data["q_table"])
        return agent

    # ------------------------------------------------------------------
    # Internal: signal extraction
    # ------------------------------------------------------------------

    def _extract_signals(
        self,
        node_features: torch.Tensor,   # (1, N, 4)  [x, y, demand/cap, is_depot]
        truck_state: torch.Tensor,      # (1, 3)     [x, y, remaining_cap/cap]
        mask: torch.Tensor,             # (1, N)     True = visited / unavailable
    ) -> tuple[dict[str, float] | None, list[int] | None]:
        """
        Compute the five geometric signals and resolve which node index each
        semantic action maps to.

        Returns (signals, action_targets) where action_targets[i] is the node
        index the environment expects for semantic action i.
        Returns (None, None) if no unvisited customer nodes remain.
        """
        nf = node_features[0]   # (N, 4)
        ts = truck_state[0]     # (3,)
        mk = mask[0]            # (N,)

        truck_xy = ts[:2]                   # (2,)
        depot_xy = nf[0, :2]               # depot is node 0

        # Unvisited customer nodes (exclude depot at index 0)
        customer_mask_valid = ~mk[1:]       # True where visitable
        valid_local = torch.where(customer_mask_valid)[0]   # indices into customer array

        if len(valid_local) == 0:
            return None, None

        n_total_customers = nf.shape[0] - 1
        n_unvisited = len(valid_local)

        unvisited_xy     = nf[1:, :2][valid_local]          # (U, 2)
        global_indices   = (valid_local + 1).tolist()        # node indices for env

        # --- dist_nearest ---
        diffs_to_truck   = unvisited_xy - truck_xy.unsqueeze(0)   # (U, 2)
        dists_to_truck   = torch.norm(diffs_to_truck, dim=-1)      # (U,)
        nearest_local    = int(torch.argmin(dists_to_truck).item())
        dist_nearest_raw = dists_to_truck[nearest_local].item()

        # Normalise by max possible distance (diagonal of unit square = sqrt(2))
        MAX_DIST = 2.0 ** 0.5
        dist_nearest = min(dist_nearest_raw / MAX_DIST, 1.0)

        # --- isolation_nearest ---
        # Isolation of the nearest node = its distance to its closest unvisited neighbour.
        # If it's the only unvisited node left, isolation = 1.0 (maximally isolated).
        nearest_xy = unvisited_xy[nearest_local]   # (2,)
        if n_unvisited == 1:
            isolation_nearest = 1.0
        else:
            diffs_among = unvisited_xy - nearest_xy.unsqueeze(0)   # (U, 2)
            dists_among = torch.norm(diffs_among, dim=-1)           # (U,)
            dists_among[nearest_local] = float("inf")               # exclude self
            isolation_raw = dists_among.min().item()
            isolation_nearest = min(isolation_raw / MAX_DIST, 1.0)

        # --- detour_nearest ---
        # How much does visiting the nearest node deviate from heading toward
        # the centroid of all remaining unvisited nodes?
        # detour = (dist_to_nearest + dist_nearest_to_centroid - dist_to_centroid)
        #           / dist_to_centroid   (clamped to [0, 1])
        centroid = unvisited_xy.mean(dim=0)                         # (2,)
        dist_to_centroid = torch.norm(truck_xy - centroid).item()

        if dist_to_centroid < 1e-6:
            detour_nearest = 0.0
        else:
            dist_nearest_to_centroid = torch.norm(nearest_xy - centroid).item()
            detour_raw = (
                dist_nearest_raw + dist_nearest_to_centroid - dist_to_centroid
            ) / dist_to_centroid
            detour_nearest = float(np.clip(detour_raw, 0.0, 1.0))

        # --- dist_to_depot ---
        dist_to_depot_raw = torch.norm(truck_xy - depot_xy).item()
        dist_to_depot = min(dist_to_depot_raw / MAX_DIST, 1.0)

        # --- unvisited_fraction ---
        unvisited_fraction = n_unvisited / n_total_customers

        signals = {
            "dist_nearest":      dist_nearest,
            "isolation_nearest": isolation_nearest,
            "detour_nearest":    detour_nearest,
            "dist_to_depot":     dist_to_depot,
            "unvisited_fraction": unvisited_fraction,
        }

        # --- resolve action targets ---
        # ACTION_VISIT_NEAREST  → node with smallest distance to truck
        # ACTION_VISIT_ISOLATED → node with largest isolation score
        # ACTION_VISIT_DETOUR   → node with smallest detour cost to centroid

        target_nearest = global_indices[nearest_local]

        if n_unvisited == 1:
            target_isolated = global_indices[0]
            target_detour   = global_indices[0]
        else:
            # Isolation score per unvisited node
            isolation_scores = []
            for i in range(n_unvisited):
                xy_i = unvisited_xy[i]
                d = torch.norm(unvisited_xy - xy_i.unsqueeze(0), dim=-1)
                d[i] = float("inf")
                isolation_scores.append(d.min().item())
            target_isolated = global_indices[int(np.argmax(isolation_scores))]

            # Detour cost per unvisited node
            if dist_to_centroid < 1e-6:
                target_detour = target_nearest
            else:
                detour_costs = []
                for i in range(n_unvisited):
                    xy_i = unvisited_xy[i]
                    d_to_node   = dists_to_truck[i].item()
                    d_node_cent = torch.norm(xy_i - centroid).item()
                    detour_costs.append(d_to_node + d_node_cent - dist_to_centroid)
                target_detour = global_indices[int(np.argmin(detour_costs))]

        action_targets = [target_nearest, target_isolated, target_detour]
        return signals, action_targets

    # ------------------------------------------------------------------
    # Internal: fuzzy state and Q aggregation
    # ------------------------------------------------------------------

    def _get_state_weights(self, signals: dict[str, float]) -> list[tuple[tuple, float]]:
        """
        Compute the cross-product of active fuzzy labels across all signals,
        weighted by the product of their memberships.
        """
        per_signal = []
        for name in SIGNAL_NAMES:
            memberships = fuzzify(signals[name], BREAKPOINTS[name])
            if not memberships:
                # Value fell outside all MF support — assign to boundary label
                memberships = {"H": 1.0} if signals[name] >= 1.0 else {"L": 1.0}
            per_signal.append(memberships)

        state_weights: list[tuple[tuple, float]] = [((), 1.0)]
        for memberships in per_signal:
            new_state_weights = []
            for partial_key, partial_weight in state_weights:
                for label, mu in memberships.items():
                    new_state_weights.append((partial_key + (label,), partial_weight * mu))
            state_weights = new_state_weights

        return state_weights

    def _aggregate_q(self, state_weights: list[tuple[tuple, float]]) -> list[float]:
        """Weighted average of Q-values across all active fuzzy state combinations."""
        total_weight = sum(w for _, w in state_weights)
        q_values = [0.0] * N_ACTIONS
        for state_key, weight in state_weights:
            norm_w = weight / total_weight
            for a in range(N_ACTIONS):
                q_values[a] += norm_w * self.q_table[state_key][a]
        return q_values

