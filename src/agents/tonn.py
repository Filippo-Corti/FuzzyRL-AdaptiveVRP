from __future__ import annotations

import torch

from src.vrp.environment_batch import VRPEnvironmentBatch


class TONNAgent:
    """
    Time-Oriented Nearest Neighbour (TONN) heuristic baseline.

    For each candidate customer j, computes:
        score_j = w_d * normalized_distance_j + w_u * urgency_j + w_f * feasibility_penalty_j

    and selects the minimum score among valid candidates.

    Notes:
    - Feasibility is primarily enforced by the env valid mask.
    - We still keep the feasibility penalty term in the score for completeness.
    - Default `w_u` is negative so high urgency lowers score (higher priority).
    """

    def __init__(
        self,
        w_d: float = 1.0,
        w_u: float = -2.0,
        w_f: float = 1.0,
        infeasible_penalty: float = 1e6,
    ):
        self.w_d = float(w_d)
        self.w_u = float(w_u)
        self.w_f = float(w_f)
        self.infeasible_penalty = float(infeasible_penalty)

    def select_actions(self, env: VRPEnvironmentBatch) -> torch.Tensor:
        """Return batched actions `(B,)` compatible with `env.execute()`/`env.solve()`."""
        obs = env.get_observation()
        valid_mask = obs["valid_action_mask"]  # (B, N+1), True=valid

        # Customer-only masks and tensors
        customer_valid = valid_mask[:, 1:]  # (B, N)
        customer_any_valid = customer_valid.any(dim=1)

        B, N = customer_valid.shape
        d = env.device

        # Distances from truck to each customer.
        truck_xy = env.truck_xy.unsqueeze(1)  # (B, 1, 2)
        customer_xy = env.instance.node_xy  # (B, N, 2)
        dists = torch.norm(customer_xy - truck_xy, dim=2)  # (B, N)

        # Normalize distance by mean feasible distance per instance.
        feasible_dist_sum = (dists * customer_valid.to(dists.dtype)).sum(dim=1, keepdim=True)
        feasible_count = customer_valid.sum(dim=1, keepdim=True).clamp(min=1)
        feasible_dist_mean = feasible_dist_sum / feasible_count
        norm_dist = dists / (feasible_dist_mean + 1e-9)

        # Urgency = time_elapsed_since_arrival / window_length.
        # time_elapsed for unseen nodes is clamped at 0; unseen nodes are masked anyway.
        t = env.timestep.unsqueeze(1)  # (B, 1)
        appearances = env.instance.appearances  # (B, N)
        window_lengths = env.instance.window_lengths.to(torch.float32)  # (B, N)
        time_elapsed = (t - appearances).clamp(min=0).to(torch.float32)
        urgency = time_elapsed / (window_lengths + 1e-9)

        # Feasibility penalty term.
        feasibility_penalty = torch.where(
            customer_valid,
            torch.zeros(B, N, device=d, dtype=torch.float32),
            torch.full((B, N), self.infeasible_penalty, device=d, dtype=torch.float32),
        )

        score = (
            self.w_d * norm_dist
            + self.w_u * urgency
            + self.w_f * feasibility_penalty
        )

        # Select minimum score among customers; fallback to depot when no valid customer.
        best_customer = torch.argmin(score, dim=1) + 1  # action indices 1..N
        depot_action = torch.zeros(B, dtype=torch.long, device=d)
        actions = torch.where(customer_any_valid, best_customer.to(torch.long), depot_action)
        return actions
