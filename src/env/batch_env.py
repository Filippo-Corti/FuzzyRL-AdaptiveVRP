import torch
from dataclasses import dataclass
from typing import Literal


@dataclass
class BatchInstanceData:
    """Static, per-instance tensors sampled at episode reset."""

    node_xy: torch.Tensor  # (B, N, 2)
    node_demands: torch.Tensor  # (B, N)
    depot_xy: torch.Tensor  # (B, 2)
    capacity: torch.Tensor  # (B,)


@dataclass
class BatchSolutionData:
    """Dynamic tensors updated at every environment step."""

    visited: torch.Tensor  # (B, N) bool
    remaining_cap: torch.Tensor  # (B,)
    truck_xy: torch.Tensor  # (B, 2)
    at_depot: torch.Tensor  # (B,) bool


@dataclass
class AgentObservation:
    """Semantic wrapper around agent-facing observation tensors."""

    node_features: torch.Tensor  # (B, N+1, 4)
    truck_state: torch.Tensor  # (B, 3)
    mask: torch.Tensor  # (B, N+1) bool

    def __iter__(self):
        yield self.node_features
        yield self.truck_state
        yield self.mask


@dataclass
class EnvActions:
    """Semantic wrapper around environment action indices."""

    indices: torch.Tensor  # (B,) int


class BatchVRPEnv:
    """
    Vectorised VRP environment for batched training.
    Operates entirely in tensor space — no VRPEnvironment, no Truck, no VRPGraph.

    Shapes throughout:
        B = batch size
        N = number of customer nodes (depot is index 0)
    """

    def __init__(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        depot_mode: Literal["center", "random"] = "center",
        node_xy_range: tuple[float, float] = (0.0, 1.0),
        demand_range: tuple[int, int] = (1, 1),
        capacity_range: tuple[int, int] = (3, 7),
    ):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.device = device
        self.depot_mode = depot_mode
        self.node_xy_range = node_xy_range
        self.demand_range = demand_range
        self.capacity_range = capacity_range

        self.static: BatchInstanceData | None = None
        self.dynamic: BatchSolutionData | None = None

    @property
    def node_xy(self) -> torch.Tensor | None:
        return None if self.static is None else self.static.node_xy

    @property
    def node_demands(self) -> torch.Tensor | None:
        return None if self.static is None else self.static.node_demands

    @property
    def depot_xy(self) -> torch.Tensor | None:
        return None if self.static is None else self.static.depot_xy

    @property
    def capacity(self) -> torch.Tensor | None:
        return None if self.static is None else self.static.capacity

    @property
    def visited(self) -> torch.Tensor | None:
        return None if self.dynamic is None else self.dynamic.visited

    @property
    def remaining_cap(self) -> torch.Tensor | None:
        return None if self.dynamic is None else self.dynamic.remaining_cap

    @property
    def truck_xy(self) -> torch.Tensor | None:
        return None if self.dynamic is None else self.dynamic.truck_xy

    @property
    def at_depot(self) -> torch.Tensor | None:
        return None if self.dynamic is None else self.dynamic.at_depot

    def reset(self):
        """Generate a new batch of random VRP instances and reset state."""
        B, N = self.batch_size, self.num_nodes
        d = self.device

        # Node positions
        xy_low, xy_high = self.node_xy_range
        node_xy = xy_low + (xy_high - xy_low) * torch.rand(B, N, 2, device=d)
        if self.depot_mode == "center":
            depot_xy = torch.tensor([0.5, 0.5], device=d).expand(B, 2)
        else:
            depot_xy = xy_low + (xy_high - xy_low) * torch.rand(B, 2, device=d)

        # Demands and capacity
        demand_min, demand_max = self.demand_range
        node_demands = torch.randint(
            demand_min, demand_max + 1, (B, N), device=d
        ).float()
        cap_min, cap_max = self.capacity_range
        capacity = torch.randint(cap_min, cap_max + 1, (B,), device=d).float()

        self.static = BatchInstanceData(
            node_xy=node_xy,
            node_demands=node_demands,
            depot_xy=depot_xy,
            capacity=capacity,
        )

        # Reset dynamic state
        self.dynamic = BatchSolutionData(
            visited=torch.zeros(B, N, dtype=torch.bool, device=d),
            remaining_cap=capacity.clone(),
            truck_xy=depot_xy.clone(),
            at_depot=torch.ones(B, dtype=torch.bool, device=d),
        )

    def _assert_ready(self) -> tuple[BatchInstanceData, BatchSolutionData]:
        assert self.static is not None
        assert self.dynamic is not None
        return self.static, self.dynamic

    def get_state(self) -> AgentObservation:
        """
        Returns the current state as tensors ready for the encoder and decoder.

        node_features: (B, N+1, 4)
            For each node: [x, y, demand/capacity, is_depot]
            Index 0 is the depot (demand=0, is_depot=1)

        truck_state: (B, 3)
            [truck_x, truck_y, remaining_cap/capacity]

        mask: (B, N+1) bool
            True = invalid action
            Depot (index 0): invalid when truck is already at depot
            Customers: invalid when visited or demand > remaining_cap
        """
        B, N = self.batch_size, self.num_nodes
        d = self.device

        static, dynamic = self._assert_ready()

        # Depot features: demand=0, is_depot=1
        depot_demand = torch.zeros(B, 1, device=d)
        depot_is_depot = torch.ones(B, 1, device=d)
        depot_features = torch.cat(
            [
                static.depot_xy.unsqueeze(1),  # (B, 1, 2)
                depot_demand.unsqueeze(-1),  # (B, 1, 1)
                depot_is_depot.unsqueeze(-1),  # (B, 1, 1)
            ],
            dim=-1,
        )  # (B, 1, 4)

        # Customer features
        demand_frac = static.node_demands / static.capacity.unsqueeze(1)  # (B, N)
        is_depot = torch.zeros(B, N, device=d)
        customer_features = torch.cat(
            [
                static.node_xy,  # (B, N, 2)
                demand_frac.unsqueeze(-1),  # (B, N, 1)
                is_depot.unsqueeze(-1),  # (B, N, 1)
            ],
            dim=-1,
        )  # (B, N, 4)

        node_features = torch.cat(
            [depot_features, customer_features], dim=1
        )  # (B, N+1, 4)

        # Truck state
        truck_state = torch.cat(
            [
                dynamic.truck_xy,  # (B, 2)
                (dynamic.remaining_cap / static.capacity).unsqueeze(1),  # (B, 1)
            ],
            dim=-1,
        )  # (B, 3)

        # Mask
        depot_mask = dynamic.at_depot.unsqueeze(1)  # (B, 1) — invalid if already at depot
        customer_mask = dynamic.visited | (
            static.node_demands > dynamic.remaining_cap.unsqueeze(1)
        )  # (B, N)
        mask = torch.cat([depot_mask, customer_mask], dim=1)  # (B, N+1)

        # Safety: if all customers are masked, force depot to be valid
        all_customers_masked = customer_mask.all(dim=1, keepdim=True)  # (B, 1)
        depot_mask = depot_mask & ~all_customers_masked
        mask = torch.cat([depot_mask, customer_mask], dim=1)

        return AgentObservation(
            node_features=node_features,
            truck_state=truck_state,
            mask=mask,
        )

    def step(self, actions: torch.Tensor | EnvActions) -> torch.Tensor:
        """
        Execute one step for all instances in the batch.

        actions: (B,) int — index into node_features (0 = depot, 1..N = customers)
        returns: (B,) float — step reward (negative distance)
        """
        static, dynamic = self._assert_ready()
        action_indices = actions.indices if isinstance(actions, EnvActions) else actions

        prev_xy = dynamic.truck_xy.clone()

        is_depot = action_indices == 0
        is_customer = ~is_depot

        # Move truck to depot
        if is_depot.any():
            dynamic.truck_xy[is_depot] = static.depot_xy[is_depot]
            dynamic.remaining_cap[is_depot] = static.capacity[is_depot]
            dynamic.at_depot[is_depot] = True

        # Move truck to customer
        if is_customer.any():
            customer_idx = action_indices[is_customer] - 1  # 0-indexed into node_xy
            dynamic.truck_xy[is_customer] = static.node_xy[is_customer, customer_idx]
            dynamic.remaining_cap[is_customer] -= static.node_demands[
                is_customer, customer_idx
            ]
            # Mark visited
            batch_idx = torch.where(is_customer)[0]
            dynamic.visited[batch_idx, customer_idx] = True
            dynamic.at_depot[is_customer] = False

        # Reward: negative distance travelled
        dist = torch.norm(dynamic.truck_xy - prev_xy, dim=-1)  # (B,)
        return -dist

    def is_done(self) -> torch.Tensor:
        """(B,) bool — True when all customers visited and truck at depot."""
        _, dynamic = self._assert_ready()
        return dynamic.visited.all(dim=1) & dynamic.at_depot

    def all_done(self) -> bool:
        return bool(self.is_done().all().item())
