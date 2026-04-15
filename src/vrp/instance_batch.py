from __future__ import annotations

from pathlib import Path
import torch
from typing import Any, Literal, cast


class VRPInstanceBatch:
    """
    Batched VRP instance generator (instance-only, no solution state).

    Tensor shapes:
            B = batch_size
            N = num_nodes (customers only)

    Public generated tensors after `generate()`:
            node_xy:         (B, N, 2)
            depot_xy:        (B, 2)
            all_xy:          (B, N+1, 2)   # [depot, customers]
            dist_matrix:     (B, N+1, N+1) # Euclidean distances on all_xy
            node_weights:    (B, N)
            W:               (B,)
            appearances:     (B, N) int timesteps
            window_lengths:  (B, N)
    """

    def __init__(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        depot_mode: Literal["center", "random"] = "center",
        node_xy_range: tuple[float, float] = (0.0, 1.0),
        weight_range: tuple[float, float] = (1.0, 5.0),
        W_value: float | None = None,
        W_range: tuple[float, float] = (20.0, 40.0),
        initial_visible_ratio: float = 0.7,
        window_length_range: tuple[int, int] = (5, 20),
        cluster_count_range: tuple[int, int] = (4, 5),
        outlier_count_range: tuple[int, int] = (1, 2),
        cluster_std_range: tuple[float, float] = (0.05, 0.14),
    ):
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.device = device
        self.depot_mode = depot_mode
        self.node_xy_range = node_xy_range
        self.weight_range = weight_range
        self.W_value = W_value
        self.W_range = W_range
        self.initial_visible_ratio = initial_visible_ratio
        self.window_length_range = window_length_range
        self.cluster_count_range = cluster_count_range
        self.outlier_count_range = outlier_count_range
        self.cluster_std_range = cluster_std_range

        self.node_xy: torch.Tensor
        self.depot_xy: torch.Tensor
        self.all_xy: torch.Tensor
        self.dist_matrix: torch.Tensor
        self.node_weights: torch.Tensor
        self.W: torch.Tensor
        self.appearances: torch.Tensor
        self.window_lengths: torch.Tensor

        self.generate()

    def generate(self) -> None:
        """(Re)generate a full batch of VRP instances."""
        B, N = self.batch_size, self.num_nodes
        d = self.device

        node_xy = torch.empty(B, N, 2, device=d)
        for b in range(B):
            node_xy[b] = self._sample_clustered_nodes_single_instance(N)

        self.node_xy = node_xy

        low, high = self.node_xy_range
        if self.depot_mode == "center":
            c = (low + high) * 0.5
            self.depot_xy = torch.tensor([c, c], device=d).expand(B, 2).clone()
        else:
            self.depot_xy = low + (high - low) * torch.rand(B, 2, device=d)

        self.all_xy = torch.cat([self.depot_xy.unsqueeze(1), self.node_xy], dim=1)
        self.dist_matrix = torch.cdist(self.all_xy, self.all_xy, p=2)

        w_low, w_high = self.weight_range
        self.node_weights = w_low + (w_high - w_low) * torch.rand(B, N, device=d)

        if self.W_value is None:
            W_low, W_high = self.W_range
            self.W = W_low + (W_high - W_low) * torch.rand(B, device=d)
        else:
            self.W = torch.full((B,), float(self.W_value), device=d)

        initial_count = max(0, min(N, int(round(self.initial_visible_ratio * N))))
        self.appearances = torch.zeros(B, N, dtype=torch.long, device=d)
        for b in range(B):
            perm = torch.randperm(N, device=d)
            late_idx = perm[initial_count:]
            if (
                late_idx.numel() > 0
            ):  # Guarantee that there won't be cases where there are no valid actions but some nodes haven't appeared yet
                raw = torch.randint(
                    1, N + 1, (late_idx.numel(),), device=d, dtype=torch.long
                )
                sorted_raw = torch.sort(raw).values
                max_safe = (
                    torch.arange(1, late_idx.numel() + 1, device=d, dtype=torch.long)
                    + initial_count
                )
                self.appearances[b, late_idx] = torch.minimum(sorted_raw, max_safe)

        win_low, win_high = self.window_length_range
        self.window_lengths = torch.randint(
            win_low,
            win_high + 1,
            (B, N),
            device=d,
            dtype=torch.long,
        )

    def extract_instance(self, index: int) -> "VRPInstance":
        """Extract one instance from the batch as a standalone VRPInstance."""
        if index < 0 or index >= self.batch_size:
            raise IndexError(
                f"index {index} out of range for batch_size={self.batch_size}"
            )
        return VRPInstance.from_batch(self, index)

    def save(self, output_path: str | Path) -> None:
        """Persist batch tensors and generation metadata to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "meta": {
                "batch_size": self.batch_size,
                "num_nodes": self.num_nodes,
                "depot_mode": self.depot_mode,
                "node_xy_range": self.node_xy_range,
                "weight_range": self.weight_range,
                "W_value": self.W_value,
                "W_range": self.W_range,
                "initial_visible_ratio": self.initial_visible_ratio,
                "window_length_range": self.window_length_range,
                "cluster_count_range": self.cluster_count_range,
                "outlier_count_range": self.outlier_count_range,
                "cluster_std_range": self.cluster_std_range,
            },
            "node_xy": self.node_xy.detach().cpu(),
            "depot_xy": self.depot_xy.detach().cpu(),
            "all_xy": self.all_xy.detach().cpu(),
            "dist_matrix": self.dist_matrix.detach().cpu(),
            "node_weights": self.node_weights.detach().cpu(),
            "W": self.W.detach().cpu(),
            "appearances": self.appearances.detach().cpu(),
            "window_lengths": self.window_lengths.detach().cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls, input_path: str | Path, device: torch.device | None = None
    ) -> "VRPInstanceBatch":
        """Load a previously saved batch from disk."""
        path = Path(input_path)
        payload = cast(dict[str, Any], torch.load(path, map_location="cpu"))
        meta = cast(dict[str, Any], payload["meta"])

        batch = cls(
            batch_size=int(meta["batch_size"]),
            num_nodes=int(meta["num_nodes"]),
            device=torch.device("cpu"),
            depot_mode=cast(Literal["center", "random"], meta["depot_mode"]),
            node_xy_range=tuple(meta["node_xy_range"]),
            weight_range=tuple(meta["weight_range"]),
            W_value=meta["W_value"],
            W_range=tuple(meta["W_range"]),
            initial_visible_ratio=float(meta["initial_visible_ratio"]),
            window_length_range=tuple(meta["window_length_range"]),
            cluster_count_range=tuple(meta["cluster_count_range"]),
            outlier_count_range=tuple(meta["outlier_count_range"]),
            cluster_std_range=tuple(meta["cluster_std_range"]),
        )

        load_device = device if device is not None else torch.device("cpu")
        batch.device = load_device
        batch.node_xy = cast(torch.Tensor, payload["node_xy"]).to(load_device)
        batch.depot_xy = cast(torch.Tensor, payload["depot_xy"]).to(load_device)
        batch.all_xy = cast(torch.Tensor, payload["all_xy"]).to(load_device)
        batch.dist_matrix = cast(torch.Tensor, payload["dist_matrix"]).to(load_device)
        batch.node_weights = cast(torch.Tensor, payload["node_weights"]).to(load_device)
        batch.W = cast(torch.Tensor, payload["W"]).to(load_device)
        batch.appearances = cast(torch.Tensor, payload["appearances"]).to(load_device)
        batch.window_lengths = cast(torch.Tensor, payload["window_lengths"]).to(
            load_device
        )
        return batch

    def _sample_clustered_nodes_single_instance(self, N: int) -> torch.Tensor:
        """Sample customer coordinates with soft clusters + explicit outliers."""
        low, high = self.node_xy_range
        span = high - low
        d = self.device

        c_min, c_max = self.cluster_count_range
        n_clusters = int(torch.randint(c_min, c_max + 1, (1,), device=d).item())

        o_min, o_max = self.outlier_count_range
        n_outliers = int(torch.randint(o_min, o_max + 1, (1,), device=d).item())
        n_outliers = max(0, min(n_outliers, N))

        n_cluster_nodes = N - n_outliers

        centers = low + span * torch.rand(n_clusters, 2, device=d)

        std_low, std_high = self.cluster_std_range
        cluster_sigmas = (
            std_low + (std_high - std_low) * torch.rand(n_clusters, 1, device=d)
        ) * span

        points: list[torch.Tensor] = []
        if n_cluster_nodes > 0:
            assignments = torch.randint(0, n_clusters, (n_cluster_nodes,), device=d)
            for idx in range(n_cluster_nodes):
                c_idx = int(assignments[idx].item())
                pt = centers[c_idx] + cluster_sigmas[c_idx] * torch.randn(2, device=d)
                points.append(torch.clamp(pt, min=low, max=high))

        if n_outliers > 0:
            if n_clusters > 0:
                min_far = cluster_sigmas.mean().item() * 2.5
            else:
                min_far = 0.2 * span

            for _ in range(n_outliers):
                accepted: torch.Tensor | None = None
                for _attempt in range(60):
                    cand = low + span * torch.rand(2, device=d)
                    dists = torch.norm(centers - cand.unsqueeze(0), dim=1)
                    if float(dists.min().item()) >= min_far:
                        accepted = cand
                        break
                if accepted is None:
                    accepted = low + span * torch.rand(2, device=d)
                points.append(accepted)

        node_xy = torch.stack(points, dim=0)
        shuffle = torch.randperm(N, device=d)
        return node_xy[shuffle]


class VRPInstance(VRPInstanceBatch):
    """Single-instance VRP wrapper with `batch_size` fixed to 1."""

    def __init__(
        self,
        num_nodes: int,
        device: torch.device,
        depot_mode: Literal["center", "random"] = "center",
        node_xy_range: tuple[float, float] = (0.0, 1.0),
        weight_range: tuple[float, float] = (1.0, 5.0),
        W_value: float | None = None,
        W_range: tuple[float, float] = (20.0, 40.0),
        initial_visible_ratio: float = 0.7,
        window_length_range: tuple[int, int] = (5, 20),
        cluster_count_range: tuple[int, int] = (4, 5),
        outlier_count_range: tuple[int, int] = (1, 2),
        cluster_std_range: tuple[float, float] = (0.05, 0.14),
    ):
        super().__init__(
            batch_size=1,
            num_nodes=num_nodes,
            device=device,
            depot_mode=depot_mode,
            node_xy_range=node_xy_range,
            weight_range=weight_range,
            W_value=W_value,
            W_range=W_range,
            initial_visible_ratio=initial_visible_ratio,
            window_length_range=window_length_range,
            cluster_count_range=cluster_count_range,
            outlier_count_range=outlier_count_range,
            cluster_std_range=cluster_std_range,
        )

    @classmethod
    def from_batch(cls, batch: VRPInstanceBatch, index: int) -> "VRPInstance":
        """Build a single instance by copying tensors from a batched container."""
        instance = cls(
            num_nodes=batch.num_nodes,
            device=batch.device,
            depot_mode=cast(Literal["center", "random"], batch.depot_mode),
            node_xy_range=batch.node_xy_range,
            weight_range=batch.weight_range,
            W_value=batch.W_value,
            W_range=batch.W_range,
            initial_visible_ratio=batch.initial_visible_ratio,
            window_length_range=batch.window_length_range,
            cluster_count_range=batch.cluster_count_range,
            outlier_count_range=batch.outlier_count_range,
            cluster_std_range=batch.cluster_std_range,
        )
        instance.node_xy = batch.node_xy[index : index + 1].clone()
        instance.depot_xy = batch.depot_xy[index : index + 1].clone()
        instance.all_xy = batch.all_xy[index : index + 1].clone()
        instance.dist_matrix = batch.dist_matrix[index : index + 1].clone()
        instance.node_weights = batch.node_weights[index : index + 1].clone()
        instance.W = batch.W[index : index + 1].clone()
        instance.appearances = batch.appearances[index : index + 1].clone()
        instance.window_lengths = batch.window_lengths[index : index + 1].clone()
        return instance
