from __future__ import annotations

import torch
from typing import Literal


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
			if late_idx.numel() > 0:
				self.appearances[b, late_idx] = torch.randint(
					1,
					N + 1,
					(late_idx.numel(),),
					device=d,
					dtype=torch.long,
				)

		win_low, win_high = self.window_length_range
		self.window_lengths = torch.randint(
			win_low,
			win_high + 1,
			(B, N),
			device=d,
			dtype=torch.long,
		)

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
		cluster_sigmas = (std_low + (std_high - std_low) * torch.rand(n_clusters, 1, device=d)) * span

		points: list[torch.Tensor] = []
		if n_cluster_nodes > 0:
			assignments = torch.randint(0, n_clusters, (n_cluster_nodes,), device=d)
			for idx in range(n_cluster_nodes):
				c_idx = int(assignments[idx].item())
				pt = centers[c_idx] + cluster_sigmas[c_idx] * torch.randn(2, device=d)
				points.append(torch.clamp(pt, min=low, max=high))

		if n_outliers > 0:
			if n_clusters > 0:
				min_far = (cluster_sigmas.mean().item() * 2.5)
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

