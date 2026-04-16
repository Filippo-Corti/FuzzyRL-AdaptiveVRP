from __future__ import annotations

from collections.abc import Iterator
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# IMPORTANT: truck_state layout assumptions.
# The FuzzyAgent indexes into truck_state with these positions:
#   [0] = current truck x position, normalised [0, 1]
#   [1] = current truck y position, normalised [0, 1]
#   [2] = remaining capacity fraction  (W_remaining / W_total), in [0, 1]
# Verify these match your VRPEnvironmentBatch.get_observation() output.
# ---------------------------------------------------------------------------

# TODO: Check these cause I think the urgency is wrong and there is no urgency in the observation? This would be a big problem for the transformer as well.
# node_features layout assumptions (same as Transformer):
#   [0] = x, [1] = y, [2] = demand (normalised), [3] = urgency,
#   [4] = visited flag, [5] = is_depot flag

_TRUCK_X_IDX = 0
_TRUCK_Y_IDX = 1
_TRUCK_CAP_IDX = 2

_NODE_DEMAND_IDX = 2
_NODE_URGENCY_IDX = 3
_NODE_VISITED_IDX = 4

CLUSTER_RADIUS = 0.2   # radius for cluster density; tune if needed
NUM_FEATURES = 5
NUM_LABELS = 3


# ---------------------------------------------------------------------------
# Triangular membership function
# ---------------------------------------------------------------------------

class TriangularMF(nn.Module):
    """
    Single triangular MF with learnable breakpoints a < b < c.

    Parameterised as:
        a = base
        b = a + softplus(d1)
        c = b + softplus(d2)
    so ordering is guaranteed and the function is differentiable everywhere.
    """

    def __init__(self, a: float, b: float, c: float) -> None:
        super().__init__()
        assert a < b < c, "Initialisation requires a < b < c"
        self._base = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        # Inverse softplus: softplus_inv(x) = log(exp(x) - 1)
        self._d1 = nn.Parameter(torch.tensor(math.log(math.expm1(b - a)), dtype=torch.float32))
        self._d2 = nn.Parameter(torch.tensor(math.log(math.expm1(c - b)), dtype=torch.float32))

    @property
    def breakpoints(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self._base
        b = a + F.softplus(self._d1)
        c = b + F.softplus(self._d2)
        return a, b, c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self.breakpoints
        left  = (x - a) / (b - a + 1e-8)
        right = (c - x) / (c - b + 1e-8)
        return torch.clamp(torch.min(left, right), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Three-label fuzzy feature (Low / Medium / High)
# ---------------------------------------------------------------------------

class FuzzyFeature(nn.Module):
    """
    Three triangular MFs for one scalar feature, covering [0, 1].

        Low    : peaks at 0,   falls to 0 at 0.5
        Medium : peaks at 0.5, zero at 0 and 1
        High   : rises from 0.5, peaks at 1
    """

    def __init__(self) -> None:
        super().__init__()
        self.low    = TriangularMF(-0.5, 0.0,  0.5)
        self.medium = TriangularMF( 0.0, 0.5,  1.0)
        self.high   = TriangularMF( 0.5, 1.0,  1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...,)  →  (..., 3)  [Low, Medium, High] membership degrees."""
        return torch.stack([self.low(x), self.medium(x), self.high(x)], dim=-1)


# ---------------------------------------------------------------------------
# Fuzzy Agent
# ---------------------------------------------------------------------------

class FuzzyAgent(nn.Module):
    """
    Scores every candidate node using a fuzzy inference system, then selects
    via softmax sampling (training) or argmax (evaluation).

    Architecture
    ────────────
    1. Compute 5 normalised scalar features per candidate node.
    2. Pass each feature through its FuzzyFeature (3 triangular MFs) →
       (B, N+1, 5, 3) activation tensor.
    3. Multiply activations by learnable rule_weights (5, 3) and sum →
       (B, N+1) priority logits.
    4. Mask invalid actions and return logits.

    Features (all normalised to [0, 1])
    ────────────────────────────────────
    0  distance       — Euclidean distance from truck to candidate / sqrt(2)
    1  urgency        — time_since_arrival / window_length  (from node_features)
    2  demand_ratio   — candidate demand / remaining truck capacity
    3  cluster_density— fraction of unvisited nodes within CLUSTER_RADIUS
    4  detour_cost    — (d_current→node + d_node→depot − d_current→depot) / mean_d, clipped [0,1]

    Rule weights are initialised so that semantically good actions score
    higher: near > far, urgent > not urgent, dense cluster > sparse, cheap
    detour > expensive, light demand > heavy demand.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.mf_distance        = FuzzyFeature()
        self.mf_urgency         = FuzzyFeature()
        self.mf_demand_ratio    = FuzzyFeature()
        self.mf_cluster_density = FuzzyFeature()
        self.mf_detour_cost     = FuzzyFeature()

        # rule_weights[f, l] = contribution of label l of feature f to priority
        # Shape: (NUM_FEATURES=5, NUM_LABELS=3)
        init = torch.tensor([
            # Low   Medium  High
            [ 0.5,   0.0,  -0.5],  # distance:        near=good, far=bad
            [-0.5,   0.0,   0.5],  # urgency:          urgent=good
            [ 0.5,   0.0,  -0.5],  # demand_ratio:     light=good (more room)
            [-0.5,   0.0,   0.5],  # cluster_density:  dense=good
            [ 0.5,   0.0,  -0.5],  # detour_cost:      cheap=good
        ], dtype=torch.float32)
        self.rule_weights = nn.Parameter(init)

        self.to(device)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        obs: dict[str, torch.Tensor],
        dist_matrix: torch.Tensor,   # (B, N+1, N+1) precomputed from instance
        depot_xy: torch.Tensor,       # (B, 2)
    ) -> torch.Tensor:
        """Returns (B, N+1, 5) normalised features."""
        node_features = obs["node_features"]   # (B, N+1, F)
        truck_state   = obs["truck_state"]     # (B, S)
        B, N1, _ = node_features.shape

        truck_xy   = truck_state[:, _TRUCK_X_IDX:_TRUCK_Y_IDX + 1]  # (B, 2)
        rem_cap    = truck_state[:, _TRUCK_CAP_IDX].unsqueeze(1)     # (B, 1)

        all_xy = torch.cat([depot_xy.unsqueeze(1),
                            node_features[:, 1:, :2]], dim=1)        # (B, N+1, 2)

        # --- 0: distance ---
        diff = all_xy - truck_xy.unsqueeze(1)                         # (B, N+1, 2)
        dist_to_nodes = torch.norm(diff, dim=-1)                      # (B, N+1)
        feat_distance = (dist_to_nodes / math.sqrt(2)).clamp(0.0, 1.0)

        # --- 1: urgency ---
        feat_urgency = node_features[..., _NODE_URGENCY_IDX].clamp(0.0, 1.0)

        # --- 2: demand ratio ---
        demands = node_features[..., _NODE_DEMAND_IDX]                # (B, N+1)
        feat_demand_ratio = (demands / (rem_cap + 1e-8)).clamp(0.0, 1.0)

        # --- 3: cluster density ---
        visited = node_features[..., _NODE_VISITED_IDX]               # (B, N+1)
        unvisited = (1.0 - visited).unsqueeze(1)                      # (B, 1, N+1)
        within = (dist_matrix < CLUSTER_RADIUS).float()               # (B, N+1, N+1)
        within = within * unvisited
        eye = torch.eye(N1, device=self.device).unsqueeze(0)
        within = within * (1.0 - eye)
        density = within.sum(dim=-1)                                  # (B, N+1)
        feat_cluster = (density / float(N1 - 1 + 1e-8)).clamp(0.0, 1.0)

        # --- 4: detour cost ---
        dist_node_depot = dist_matrix[:, :, 0]                        # (B, N+1)
        dist_truck_depot = torch.norm(
            truck_xy - depot_xy, dim=-1
        ).unsqueeze(1)                                                 # (B, 1)
        detour = dist_to_nodes + dist_node_depot - dist_truck_depot    # (B, N+1)
        mean_d = dist_matrix.mean(dim=(1, 2), keepdim=False
                                  ).mean().clamp(min=1e-8)
        feat_detour = (detour / (2.0 * mean_d)).clamp(0.0, 1.0)

        return torch.stack([
            feat_distance,
            feat_urgency,
            feat_demand_ratio,
            feat_cluster,
            feat_detour,
        ], dim=-1)  # (B, N+1, 5)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        dist_matrix: torch.Tensor,
        depot_xy: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns (B, N+1) logits. Invalid positions are set to -inf.
        """
        features = self._compute_features(obs, dist_matrix, depot_xy)  # (B, N+1, 5)

        acts = torch.stack([
            self.mf_distance(features[..., 0]),
            self.mf_urgency(features[..., 1]),
            self.mf_demand_ratio(features[..., 2]),
            self.mf_cluster_density(features[..., 3]),
            self.mf_detour_cost(features[..., 4]),
        ], dim=2)  # (B, N+1, 5, 3)

        weighted = acts * self.rule_weights.unsqueeze(0).unsqueeze(0)   # (B, N+1, 5, 3)
        logits = weighted.sum(dim=(-1, -2))                             # (B, N+1)
        logits = logits.masked_fill(invalid_mask, float("-inf"))
        return logits

    # ------------------------------------------------------------------
    # Inference helper (compatible with env.solve() interface)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(self, env, greedy: bool = True) -> torch.Tensor:
        obs = env.get_observation()
        logits = self.forward(
            obs=obs,
            dist_matrix=env.instance_batch.dist_matrix,
            depot_xy=env.instance_batch.depot_xy,
            invalid_mask=obs["invalid_action_mask"],
        )
        if greedy:
            return logits.argmax(dim=-1).to(torch.long)
        return torch.distributions.Categorical(logits=logits).sample().to(torch.long)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "FuzzyAgent":
        agent = cls(device=device)
        agent.load_state_dict(torch.load(path, map_location=device))
        return agent

    # ------------------------------------------------------------------
    # Introspection helpers for the demo / report
    # ------------------------------------------------------------------

    def top_rules(self, top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        Return top-k (feature, label, weight) triples by absolute weight.
        Useful for the rule display panel in the pygame demo.
        """
        feature_names = ["distance", "urgency", "demand_ratio", "cluster_density", "detour_cost"]
        label_names   = ["low", "medium", "high"]
        w = self.rule_weights.detach().cpu()
        entries = [
            (feature_names[f], label_names[l], float(w[f, l]))
            for f in range(NUM_FEATURES)
            for l in range(NUM_LABELS)
        ]
        return sorted(entries, key=lambda t: abs(t[2]), reverse=True)[:top_k]