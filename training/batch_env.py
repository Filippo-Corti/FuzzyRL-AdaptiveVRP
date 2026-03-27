import torch
import torch.nn.functional as F


class BatchVRPEnv:
    """
    Vectorised VRP environment for batched training.
    Operates entirely in tensor space — no VRPEnvironment, no Truck, no VRPGraph.

    Shapes throughout:
        B = batch size
        N = number of customer nodes (depot is index 0)
    """

    def __init__(self, batch_size: int, num_nodes: int, device: torch.device):
        self.B = batch_size
        self.N = num_nodes  # customers only, depot handled separately
        self.device = device

        # Generated fresh each call to reset()
        self.node_xy: torch.Tensor = None        # (B, N, 2)
        self.node_demands: torch.Tensor = None   # (B, N)
        self.depot_xy: torch.Tensor = None       # (B, 2)
        self.capacity: torch.Tensor = None       # (B,)

        # Dynamic state
        self.visited: torch.Tensor = None        # (B, N) bool
        self.remaining_cap: torch.Tensor = None  # (B,)
        self.truck_xy: torch.Tensor = None       # (B, 2)
        self.at_depot: torch.Tensor = None       # (B,) bool

    def reset(self):
        """Generate a new batch of random VRP instances and reset state."""
        B, N = self.B, self.N
        d = self.device

        # Node positions: uniform [0, 1]
        self.depot_xy = torch.rand(B, 2, device=d)
        self.node_xy = torch.rand(B, N, 2, device=d)

        # Demands: uniform int [1, 10]
        self.node_demands = torch.randint(1, 11, (B, N), device=d).float()

        # Capacity: enough for 4-10 nodes on average (mean demand 5.5)
        min_cap = int(5.5 * 4)
        max_cap = int(5.5 * 10)
        self.capacity = torch.randint(
            min_cap, max_cap + 1, (B,), device=d
        ).float()

        # Reset dynamic state
        self.visited = torch.zeros(B, N, dtype=torch.bool, device=d)
        self.remaining_cap = self.capacity.clone()
        self.truck_xy = self.depot_xy.clone()
        self.at_depot = torch.ones(B, dtype=torch.bool, device=d)

    def get_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        B, N = self.B, self.N

        # Depot features: demand=0, is_depot=1
        depot_demand = torch.zeros(B, 1, device=self.device)
        depot_is_depot = torch.ones(B, 1, device=self.device)
        depot_features = torch.cat([
            self.depot_xy.unsqueeze(1),          # (B, 1, 2)
            depot_demand.unsqueeze(-1),           # (B, 1, 1)
            depot_is_depot.unsqueeze(-1),         # (B, 1, 1)
        ], dim=-1)  # (B, 1, 4)

        # Customer features
        demand_frac = self.node_demands / self.capacity.unsqueeze(1)  # (B, N)
        is_depot = torch.zeros(B, N, device=self.device)
        customer_features = torch.cat([
            self.node_xy,                         # (B, N, 2)
            demand_frac.unsqueeze(-1),            # (B, N, 1)
            is_depot.unsqueeze(-1),               # (B, N, 1)
        ], dim=-1)  # (B, N, 4)

        node_features = torch.cat([depot_features, customer_features], dim=1)  # (B, N+1, 4)

        # Truck state
        truck_state = torch.cat([
            self.truck_xy,                                              # (B, 2)
            (self.remaining_cap / self.capacity).unsqueeze(1),         # (B, 1)
        ], dim=-1)  # (B, 3)

        # Mask
        depot_mask = self.at_depot.unsqueeze(1)  # (B, 1) — invalid if already at depot
        customer_mask = self.visited | (
            self.node_demands > self.remaining_cap.unsqueeze(1)
        )  # (B, N)
        mask = torch.cat([depot_mask, customer_mask], dim=1)  # (B, N+1)

        # Safety: if all customers are masked, force depot to be valid
        all_customers_masked = customer_mask.all(dim=1, keepdim=True)  # (B, 1)
        depot_mask = depot_mask & ~all_customers_masked
        mask = torch.cat([depot_mask, customer_mask], dim=1)

        return node_features, truck_state, mask

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Execute one step for all instances in the batch.

        actions: (B,) int — index into node_features (0 = depot, 1..N = customers)
        returns: (B,) float — step reward (negative distance)
        """
        B = self.B
        prev_xy = self.truck_xy.clone()

        is_depot = (actions == 0)
        is_customer = ~is_depot

        # Move truck to depot
        if is_depot.any():
            self.truck_xy[is_depot] = self.depot_xy[is_depot]
            self.remaining_cap[is_depot] = self.capacity[is_depot]
            self.at_depot[is_depot] = True

        # Move truck to customer
        if is_customer.any():
            customer_idx = actions[is_customer] - 1  # 0-indexed into node_xy
            self.truck_xy[is_customer] = self.node_xy[is_customer, customer_idx]
            self.remaining_cap[is_customer] -= self.node_demands[
                is_customer, customer_idx
            ]
            # Mark visited
            batch_idx = torch.where(is_customer)[0]
            self.visited[batch_idx, customer_idx] = True
            self.at_depot[is_customer] = False

        # Reward: negative distance travelled
        dist = torch.norm(self.truck_xy - prev_xy, dim=-1)  # (B,)
        return -dist

    def is_done(self) -> torch.Tensor:
        """(B,) bool — True when all customers visited and truck at depot."""
        return self.visited.all(dim=1) & self.at_depot

    def all_done(self) -> bool:
        return self.is_done().all().item()