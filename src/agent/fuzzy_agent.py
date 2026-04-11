from __future__ import annotations

import pickle
import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

from .base import BaseAgent

# ------------------------------------------------------------------
# Membership functions
# ------------------------------------------------------------------


def _triangular(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function with peak at b, zero at a and c."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


def fuzzify(
    value: float, breakpoints: list[tuple[str, float, float, float]]
) -> dict[str, float]:
    """
    Fuzzify a crisp value using a list of (label, a, b, c) triangular MFs.
    Returns {label: membership} for all non-zero memberships.
    """
    result = {}
    for label, a, b, c in breakpoints:
        mu = _triangular(value, a, b, c)
        if mu > 0.0:
            result[label] = mu
    return result


# Membership function breakpoints for each signal.
# Each signal is mapped to Low / Medium / High with overlapping triangles.
BREAKPOINTS = {
    "dist_nearest": [("L", 0.0, 0.0, 0.3), ("M", 0.1, 0.3, 0.6), ("H", 0.4, 1.0, 1.0)],
    "demand_nearest": [
        ("L", 0.0, 0.0, 0.4),
        ("M", 0.2, 0.5, 0.8),
        ("H", 0.6, 1.0, 1.0),
    ],
    "dist_2nd": [("L", 0.0, 0.0, 0.3), ("M", 0.1, 0.3, 0.6), ("H", 0.4, 1.0, 1.0)],
    "demand_2nd": [("L", 0.0, 0.0, 0.4), ("M", 0.2, 0.5, 0.8), ("H", 0.6, 1.0, 1.0)],
    "dist_3rd": [("L", 0.0, 0.0, 0.3), ("M", 0.1, 0.3, 0.6), ("H", 0.4, 1.0, 1.0)],
    "demand_3rd": [("L", 0.0, 0.0, 0.4), ("M", 0.2, 0.5, 0.8), ("H", 0.6, 1.0, 1.0)],
    "remaining_cap": [("L", 0.0, 0.0, 0.4), ("M", 0.2, 0.5, 0.8), ("H", 0.6, 1.0, 1.0)],
}

SIGNAL_NAMES = list(BREAKPOINTS.keys())  # fixed order for state tuple construction
N_ACTIONS = 4  # nearest, 2nd nearest, 3rd nearest, depot


# ------------------------------------------------------------------
# FuzzyAgent
# ------------------------------------------------------------------


class FuzzyAgent(BaseAgent):
    """
    Fuzzy Q-learning agent for VRP.

    State: fuzzified signals from the 3 nearest unvisited nodes + remaining capacity.
    Actions: go to nearest (0), 2nd nearest (1), 3rd nearest (2), depot (3).

    The agent extracts its own features from the raw (node_features, truck_state, mask)
    tensors so it plugs into the same interface as TransformerAgent.

    Q-table is indexed by tuples of fuzzy label combinations, weighted by
    product of membership values (fuzzy inference).
    """

    def __init__(
        self,
        epsilon: float = 0.9,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 0.1,
        gamma: float = 0.95,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.gamma = gamma

        # Q[state_label_tuple][action] -> float
        self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * N_ACTIONS)

        # Last transition for Q-update
        self._last_state_weights: list[tuple[tuple, float]] | None = None
        self._last_action: int | None = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        node_features: torch.Tensor,  # (1, N+1, 4)
        truck_state: torch.Tensor,  # (1, 3)
        mask: torch.Tensor,  # (1, N+1) bool
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signals, candidates = self._extract_signals(node_features, truck_state, mask)
        if signals is None:
            # All customers masked and depot is the only option
            self._last_state_weights = None
            self._last_action = 0
            return torch.tensor([0]), torch.zeros(1)

        assert candidates is not None

        state_weights = self._get_state_weights(signals)
        q_values = self._aggregate_q(state_weights)

        # Build valid action set: indices into candidates list + depot (always last)
        valid_actions = list(range(len(candidates))) + [N_ACTIONS - 1]
        # Mask depot if at depot already (mask[0, 0] is True when depot invalid)
        if mask[0, 0].item():
            valid_actions = [a for a in valid_actions if a != N_ACTIONS - 1]
        if not valid_actions:
            valid_actions = [N_ACTIONS - 1]

        if not greedy and np.random.random() < self.epsilon:
            fuzzy_action = np.random.choice(valid_actions)
        else:
            fuzzy_action = max(valid_actions, key=lambda a: q_values[a])

        self._last_state_weights = state_weights
        self._last_action = fuzzy_action

        # Map fuzzy action index → env action index
        env_action = self._fuzzy_action_to_env(fuzzy_action, candidates, mask)
        return torch.tensor([env_action]), torch.zeros(1)

    def q_update(
        self,
        reward: float,
        next_node_features: torch.Tensor,
        next_truck_state: torch.Tensor,
        next_mask: torch.Tensor,
        done: bool,
    ):
        """Perform a Q-learning update for the last transition."""
        if self._last_state_weights is None:
            return
        assert self._last_action is not None

        # Next state Q-values
        if done:
            target = reward
        else:
            next_signals, next_candidates = self._extract_signals(
                next_node_features, next_truck_state, next_mask
            )
            if next_signals is None:
                target = reward
            else:
                next_weights = self._get_state_weights(next_signals)
                next_q = self._aggregate_q(next_weights)
                target = reward + self.gamma * max(next_q)

        # Update all activated state entries weighted by their membership
        for state_key, weight in self._last_state_weights:
            current_q = self.q_table[state_key][self._last_action]
            self.q_table[state_key][self._last_action] += (
                self.lr * weight * (target - current_q)
            )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "q_table": dict(self.q_table),
                    "epsilon": self.epsilon,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "FuzzyAgent":
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls()
        agent.q_table = defaultdict(lambda: [0.0] * N_ACTIONS, data["q_table"])
        agent.epsilon = data["epsilon"]
        return agent

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_signals(
        self,
        node_features: torch.Tensor,  # (1, N+1, 4)
        truck_state: torch.Tensor,  # (1, 3)
        mask: torch.Tensor,  # (1, N+1) bool
    ) -> tuple[dict[str, float] | None, list[int] | None]:
        """
        Extract the 7 fuzzy signals from raw tensors.

        Returns (signals_dict, candidate_env_indices) where candidate_env_indices
        are the env action indices (1-based) of the 3 nearest unvisited nodes.
        Returns (None, None) if no unvisited nodes are available.
        """
        # node_features[0, 1:, :] are customers: [x, y, demand_frac, 0]
        # truck_state[0] = [truck_x, truck_y, remaining_cap_frac]
        nf = node_features[0]  # (N+1, 4)
        ts = truck_state[0]  # (3,)
        mk = mask[0]  # (N+1,) True=invalid

        truck_xy = ts[:2]  # (2,)
        remaining_cap = ts[2].item()

        # Customer positions and validity (indices 1..N in node_features)
        customer_xy = nf[1:, :2]  # (N, 2)
        customer_valid = ~mk[1:]  # (N,) True=valid

        valid_indices = torch.where(customer_valid)[0]  # 0-based into customers
        if len(valid_indices) == 0:
            return None, None

        # Distances from truck to all valid customers
        diffs = customer_xy[valid_indices] - truck_xy.unsqueeze(0)  # (K, 2)
        dists = torch.norm(diffs, dim=-1)  # (K,)

        # Sort by distance, take up to 3
        sorted_order = torch.argsort(dists)
        top_k = sorted_order[:3]
        top_indices = valid_indices[top_k]  # 0-based customer indices
        top_dists = dists[top_k]
        top_demands = nf[1:, 2][top_indices]  # demand_frac

        # Pad to 3 if fewer than 3 valid customers
        pad = 3 - len(top_indices)
        if pad > 0:
            top_dists = torch.cat([top_dists, torch.ones(pad)])
            top_demands = torch.cat([top_demands, torch.zeros(pad)])

        signals = {
            "dist_nearest": top_dists[0].item(),
            "demand_nearest": top_demands[0].item(),
            "dist_2nd": top_dists[1].item(),
            "demand_2nd": top_demands[1].item(),
            "dist_3rd": top_dists[2].item(),
            "demand_3rd": top_demands[2].item(),
            "remaining_cap": remaining_cap,
        }

        # env action indices are 1-based (0 = depot)
        candidates = [int(idx.item()) + 1 for idx in top_indices]
        return signals, candidates

    def _get_state_weights(self, signals: dict) -> list[tuple[tuple, float]]:
        """
        Fuzzify all signals and return a list of (state_key, combined_weight) pairs.
        Each state_key is a tuple of (label, label, ...) for each signal in order.
        Combined weight is the product of individual memberships (minimum T-norm
        would also work; product gives softer boundaries).
        """
        per_signal = []
        for name in SIGNAL_NAMES:
            memberships = fuzzify(signals[name], BREAKPOINTS[name])
            if not memberships:
                # Clamp to nearest boundary label
                memberships = {"H": 1.0} if signals[name] >= 1.0 else {"L": 1.0}
            per_signal.append(memberships)

        # Cartesian product of activated labels, weighted by product of memberships
        state_weights: list[tuple[dict[int, str], float]] = [({}, 1.0)]
        for memberships in per_signal:
            new_state_weights = []
            for partial_key, partial_weight in state_weights:
                for label, mu in memberships.items():
                    new_key = {**partial_key, len(partial_key): label}
                    new_state_weights.append((new_key, partial_weight * mu))
            state_weights = new_state_weights

        return [(tuple(d.values()), w) for d, w in state_weights]

    def _aggregate_q(self, state_weights: list[tuple[tuple, float]]) -> list[float]:
        """Compute weighted average Q-values across all activated fuzzy states."""
        total_weight = sum(w for _, w in state_weights)
        q_values = [0.0] * N_ACTIONS
        for state_key, weight in state_weights:
            for a in range(N_ACTIONS):
                q_values[a] += (weight / total_weight) * self.q_table[state_key][a]
        return q_values

    def _fuzzy_action_to_env(
        self, fuzzy_action: int, candidates: list[int], mask: torch.Tensor
    ) -> int:
        """
        Map a fuzzy action index to an env action index.
        fuzzy_action 0,1,2 → candidates[0,1,2] (nearest, 2nd, 3rd)
        fuzzy_action 3     → depot (0)
        Falls back to depot if candidate index is out of range.
        """
        if fuzzy_action == N_ACTIONS - 1:
            return 0  # depot
        if fuzzy_action < len(candidates):
            return candidates[fuzzy_action]
        return 0  # fallback to depot
