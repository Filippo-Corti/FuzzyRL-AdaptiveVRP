import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    """
    Two-step attention decoder following Kool et al. (2019).

    Step 1 — Context attention:
        The truck state is projected to a query vector which attends over all
        node embeddings. This produces a context vector summarising what is
        currently relevant given the truck's position, load, and the graph.

    Step 2 — Glimpse + scoring:
        The context vector attends over node embeddings again (the "glimpse")
        to produce refined logits. Logits are tanh-clipped to [-C, C] to
        prevent the distribution from collapsing to near-deterministic early
        in training, preserving exploration.

    Input:
        node_embeddings: (B, N, d_model)  — from Encoder
        truck_state:     (B, state_features)
        mask:            (B, N) bool — True = invalid action

    Output:
        logits: (B, N) — raw (unmasked) scores; Categorical handles softmax
    """

    CLIP_C = 10.0  # tanh clipping constant from Kool et al.

    def __init__(self, state_features: int, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Project truck state to initial query
        self.query_proj = nn.Linear(state_features, d_model)

        # Step 1: context attention projections
        self.context_q = nn.Linear(d_model, d_model)
        self.context_k = nn.Linear(d_model, d_model)
        self.context_v = nn.Linear(d_model, d_model)
        self.context_out = nn.Linear(d_model, d_model)

        # Step 2: glimpse projections (single-head for final scoring)
        self.glimpse_q = nn.Linear(d_model, d_model)
        self.glimpse_k = nn.Linear(d_model, d_model)

    def forward(
        self,
        node_embeddings: torch.Tensor,  # (B, N, d_model)
        truck_state: torch.Tensor,  # (B, state_features)
        mask: torch.Tensor,  # (B, N) bool
    ) -> torch.Tensor:

        B, N, _ = node_embeddings.shape

        # --- Initial query from truck state ---
        query = self.query_proj(truck_state)  # (B, d_model)

        # --- Step 1: multi-head context attention ---
        # Query attends over all node embeddings to build a context vector.
        # This is standard multi-head attention with the truck state as query
        # and node embeddings as keys/values.

        q = self._split_heads(self.context_q(query.unsqueeze(1)))  # (B, H, 1, head_dim)
        k = self._split_heads(self.context_k(node_embeddings))  # (B, H, N, head_dim)
        v = self._split_heads(self.context_v(node_embeddings))  # (B, H, N, head_dim)

        # Mask: (B, 1, 1, N) — broadcast across heads
        attn_mask = mask.unsqueeze(1).unsqueeze(2).float() * -1e9

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, 1, N)
        attn = attn + attn_mask
        attn = torch.softmax(attn, dim=-1)  # (B, H, 1, N)

        context = (attn @ v).squeeze(2)  # (B, H, head_dim)
        context = context.reshape(B, self.d_model)  # (B, d_model)
        context = self.context_out(context)  # (B, d_model)

        # --- Step 2: glimpse — score each node against context ---
        # Single-head dot-product attention for final logits.
        # tanh clipping keeps logits in [-C, C], preventing entropy collapse.

        gq = self.glimpse_q(context)  # (B, d_model)
        gk = self.glimpse_k(node_embeddings)  # (B, N, d_model)

        logits = (gk @ gq.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_model)  # (B, N)

        # tanh clip
        logits = self.CLIP_C * torch.tanh(logits)

        # Mask invalid actions
        logits = logits.masked_fill(mask, float("-inf"))

        return logits

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, d_model) → (B, H, S, head_dim)"""
        B, S, _ = x.shape
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
