import torch
import torch.nn as nn


class Encoder(nn.Module):
	"""
	Projects node features into d_model, then runs L transformer encoder layers
	so each node embedding becomes context-aware (attends over all other nodes).

	Input:  (B, N, node_features)
	Output: (B, N, d_model)
	"""

	def __init__(
		self,
		node_features: int,
		d_model: int,
		num_layers: int = 3,
		num_heads: int = 8,
		dim_feedforward: int = 512,
		dropout: float = 0.0,
	):
		super().__init__()

		self.input_proj = nn.Linear(node_features, d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=num_heads,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
			norm_first=True,
		)
		self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

	def forward(self, nodes: torch.Tensor) -> torch.Tensor:
		x = self.input_proj(nodes)
		return self.layers(x)


__all__ = ["Encoder"]
