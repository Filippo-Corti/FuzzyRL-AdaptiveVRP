from __future__ import annotations

from typing import Callable

import torch

from .instance_batch import VRPInstanceBatch


class VRPEnvironmentBatch:
    """
    Batched VRP environment state built on top of a `VRPInstanceBatch`.

    Action convention:
        - 0: go to depot
        - 1..N: go to customer node index (1 maps to customer 0)

    Public solution-state attributes:
        visited:         (B, N) bool
        truck_xy:        (B, 2)
        at_depot:        (B,) bool
        remaining_cap:   (B,)
        timestep:        (B,) long
        done:            (B,) bool
        routes:          list[list[tuple[float, float]]] route history per instance
    """

    def __init__(
        self,
        instance_batch: VRPInstanceBatch,
        lateness_penalty_alpha: float = 1.0,
    ):
        self.instance = instance_batch
        self.device = instance_batch.device
        self.batch_size = instance_batch.batch_size
        self.num_nodes = instance_batch.num_nodes
        self.lateness_penalty_alpha = float(lateness_penalty_alpha)

        self.visited: torch.Tensor
        self.truck_xy: torch.Tensor
        self.at_depot: torch.Tensor
        self.remaining_cap: torch.Tensor
        self.timestep: torch.Tensor
        self.done: torch.Tensor
        self.total_distance: torch.Tensor
        self.total_lateness: torch.Tensor
        self.routes: list[list[tuple[float, float]]]

        self.reset(regenerate_instance=False)

    def reset(self, regenerate_instance: bool = False) -> None:
        """Reset solution state; optionally regenerate underlying instances."""
        if regenerate_instance:
            self.instance.generate()

        B, N = self.batch_size, self.num_nodes
        d = self.device

        self.visited = torch.zeros(B, N, dtype=torch.bool, device=d)
        self.truck_xy = self.instance.depot_xy.clone()
        self.at_depot = torch.ones(B, dtype=torch.bool, device=d)
        self.remaining_cap = self.instance.W.clone()
        self.timestep = torch.zeros(B, dtype=torch.long, device=d)
        self.done = torch.zeros(B, dtype=torch.bool, device=d)
        self.total_distance = torch.zeros(B, dtype=torch.float32, device=d)
        self.total_lateness = torch.zeros(B, dtype=torch.float32, device=d)

        self.routes = []
        for b in range(B):
            depot = self.instance.depot_xy[b]
            self.routes.append([(float(depot[0].item()), float(depot[1].item()))])

        self._refresh_done()

    def available_nodes_mask(self) -> torch.Tensor:
        """(B, N) bool mask for currently available customers (appeared by current timestep)."""
        return self.instance.appearances <= self.timestep.unsqueeze(1)

    def valid_action_mask(self) -> torch.Tensor:
        """
        (B, N+1) bool where True means action is valid.

        Depot action (index 0) is always valid.
        Customer action j+1 is valid if:
            - customer j has appeared
            - customer j is not visited
            - customer j weight <= remaining capacity
        """
        available = self.available_nodes_mask()
        can_serve = self.instance.node_weights <= self.remaining_cap.unsqueeze(1)
        customer_valid = available & (~self.visited) & can_serve
        depot_valid = torch.ones(self.batch_size, 1, dtype=torch.bool, device=self.device)
        return torch.cat([depot_valid, customer_valid], dim=1)

    def get_observation(self) -> dict[str, torch.Tensor]:
        """
        Build a callback-friendly batched observation on the current device.

        Returns keys:
            node_features:     (B, N+1, 6) = [x, y, weight_norm, appeared, visited, is_depot]
            truck_state:       (B, 4)      = [truck_x, truck_y, remaining_cap_norm, at_depot]
            valid_action_mask: (B, N+1) bool  (True = valid)
            invalid_action_mask:(B, N+1) bool (True = invalid)
            timestep:          (B,)
            done:              (B,)
        """
        B = self.batch_size

        # Depot row
        depot_xy = self.instance.depot_xy.unsqueeze(1)  # (B, 1, 2)
        depot_weight = torch.zeros(B, 1, 1, device=self.device)
        depot_appeared = torch.ones(B, 1, 1, device=self.device)
        depot_visited = torch.ones(B, 1, 1, device=self.device)
        depot_is_depot = torch.ones(B, 1, 1, device=self.device)
        depot_features = torch.cat(
            [
                depot_xy,
                depot_weight,
                depot_appeared,
                depot_visited,
                depot_is_depot,
            ],
            dim=-1,
        )  # (B, 1, 6)

        # Customer rows
        appeared = self.available_nodes_mask().unsqueeze(-1).to(torch.float32)  # (B, N, 1)
        visited = self.visited.unsqueeze(-1).to(torch.float32)  # (B, N, 1)
        is_depot = torch.zeros(B, self.num_nodes, 1, device=self.device)
        weight_norm = self.instance.node_weights / self.instance.W.unsqueeze(1)
        customer_features = torch.cat(
            [
                self.instance.node_xy,
                weight_norm.unsqueeze(-1),
                appeared,
                visited,
                is_depot,
            ],
            dim=-1,
        )  # (B, N, 6)

        node_features = torch.cat([depot_features, customer_features], dim=1)  # (B, N+1, 6)

        truck_state = torch.cat(
            [
                self.truck_xy,
                (self.remaining_cap / self.instance.W).unsqueeze(1),
                self.at_depot.unsqueeze(1).to(torch.float32),
            ],
            dim=1,
        )  # (B, 4)

        valid_mask = self.valid_action_mask()

        return {
            "node_features": node_features,
            "truck_state": truck_state,
            "valid_action_mask": valid_mask,
            "invalid_action_mask": ~valid_mask,
            "timestep": self.timestep,
            "done": self.done,
        }

    def execute(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Execute one batched step.

        Args:
            actions: (B,) long action indices in [0, N].

        Returns:
            step_distance: (B,) float distance moved in this step.
            step_delay:    (B,) float service lateness for the served customer,
                           or 0 for depot/no-lateness.
        """
        if actions.dim() != 1 or actions.shape[0] != self.batch_size:
            raise ValueError("actions must have shape (batch_size,)")

        done_at_start = self.done.clone()
        prev_xy = self.truck_xy.clone()

        actions = actions.to(self.device, dtype=torch.long)
        valid_mask = self.valid_action_mask()
        batch_idx = torch.arange(self.batch_size, device=self.device)

        in_bounds = (actions >= 0) & (actions <= self.num_nodes)
        chosen_valid = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        chosen_valid[in_bounds] = valid_mask[batch_idx[in_bounds], actions[in_bounds]]

        # Invalid actions fallback to depot.
        actions = torch.where(chosen_valid, actions, torch.zeros_like(actions))

        go_depot = actions == 0
        go_customer = ~go_depot

        if go_depot.any():
            self.truck_xy[go_depot] = self.instance.depot_xy[go_depot]
            self.remaining_cap[go_depot] = self.instance.W[go_depot]
            self.at_depot[go_depot] = True

        if go_customer.any():
            customer_idx = actions[go_customer] - 1
            selected_xy = self.instance.node_xy[go_customer, customer_idx]
            selected_weights = self.instance.node_weights[go_customer, customer_idx]

            self.truck_xy[go_customer] = selected_xy
            self.remaining_cap[go_customer] = self.remaining_cap[go_customer] - selected_weights
            self.at_depot[go_customer] = False

            row_idx = torch.where(go_customer)[0]
            self.visited[row_idx, customer_idx] = True

            # Delay for newly served nodes: max(0, service_t - (appearance + window_length)).
            service_t = self.timestep[row_idx]
            deadlines = (
                self.instance.appearances[row_idx, customer_idx]
                + self.instance.window_lengths[row_idx, customer_idx]
            )
            raw_delay = service_t - deadlines
            step_delay = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
            step_delay[row_idx] = torch.clamp(raw_delay.to(torch.float32), min=0.0)
        else:
            step_delay = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        # Advance time for unfinished instances only.
        self.timestep = torch.where(self.done, self.timestep, self.timestep + 1)

        # Distance travelled this step.
        step_distance = torch.norm(self.truck_xy - prev_xy, dim=1)
        step_distance = torch.where(done_at_start, torch.zeros_like(step_distance), step_distance)
        step_delay = torch.where(done_at_start, torch.zeros_like(step_delay), step_delay)

        self.total_distance = self.total_distance + step_distance
        self.total_lateness = self.total_lateness + step_delay

        # Update route traces (for visualization/debug).
        for b in range(self.batch_size):
            if bool(self.done[b].item()):
                continue
            pos = self.truck_xy[b]
            self.routes[b].append((float(pos[0].item()), float(pos[1].item())))

        self._refresh_done()
        return step_distance, step_delay

    def solve(
        self,
        select_action_callback: Callable[["VRPEnvironmentBatch"], torch.Tensor],
        max_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Roll out policy callback until all instances are finished.

        The callback receives this environment and should return a tensor of shape
        (B,) with action indices. For GPU efficiency, tensors remain on device and
        no host sync is required except route-history bookkeeping.

        Returns:
            total_cost: (B,) tensor with
                C = total_travel_distance + alpha * total_lateness
        """
        if max_steps is None:
            max_steps = max(4 * self.num_nodes, self.num_nodes + 1)

        steps = 0
        while not bool(self.done.all().item()) and steps < max_steps:
            actions = select_action_callback(self)
            self.execute(actions)
            steps += 1
        return self.total_distance + self.lateness_penalty_alpha * self.total_lateness

    def _refresh_done(self) -> None:
        """An instance is done only when every customer has been visited."""
        self.done = self.visited.all(dim=1)
