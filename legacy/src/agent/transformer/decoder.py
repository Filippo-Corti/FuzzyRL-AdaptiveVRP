import math

import torch
import torch.nn as nn


class Decoder(nn.Module):
	"""Two-step attention decoder for action logits."""

	# We intentionally keep a custom decoder block: the policy decoder needs
	# explicit masking and a two-stage glimpse/scoring flow rather than a
	# generic autoregressive TransformerDecoderLayer.
	CLIP_C = 10.0

	def __init__(self, state_features: int, d_model: int, num_heads: int = 8):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

		self.query_proj = nn.Linear(state_features, d_model)

		self.context_q = nn.Linear(d_model, d_model)
		self.context_k = nn.Linear(d_model, d_model)
		self.context_v = nn.Linear(d_model, d_model)
		self.context_out = nn.Linear(d_model, d_model)

		self.glimpse_q = nn.Linear(d_model, d_model)
		self.glimpse_k = nn.Linear(d_model, d_model)

	def forward(
		self,
		node_embeddings: torch.Tensor,
		truck_state: torch.Tensor,
		mask: torch.Tensor,
	) -> torch.Tensor:
		B, N, _ = node_embeddings.shape
		query = self.query_proj(truck_state)

		q = self._split_heads(self.context_q(query.unsqueeze(1)))
		k = self._split_heads(self.context_k(node_embeddings))
		v = self._split_heads(self.context_v(node_embeddings))

		attn_mask = mask.unsqueeze(1).unsqueeze(2).float() * -1e9
		attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
		attn = attn + attn_mask
		attn = torch.softmax(attn, dim=-1)

		context = (attn @ v).squeeze(2)
		context = context.reshape(B, self.d_model)
		context = self.context_out(context)

		gq = self.glimpse_q(context)
		gk = self.glimpse_k(node_embeddings)

		logits = (gk @ gq.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_model)
		# tanh clipping follows Kool et al.; it prevents early extreme logits.
		logits = self.CLIP_C * torch.tanh(logits)
		logits = logits.masked_fill(mask, float("-inf"))
		return logits

	def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
		B, S, _ = x.shape
		return x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)


__all__ = ["Decoder"]
