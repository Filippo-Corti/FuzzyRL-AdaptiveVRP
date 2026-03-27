import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    """
    Attention decoder: builds a context vector from truck state attending over
    node embeddings, then scores each node against that context.
    """

    def __init__(self, state_features: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(state_features, d_model)
        self.scores_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        node_embeddings: torch.Tensor,  # (B, N, d_model)
        truck_state: torch.Tensor,       # (B, state_features)
        mask: torch.Tensor,              # (B, N) True = invalid
    ) -> torch.Tensor:
        # Query from truck state
        query = self.query_proj(truck_state).unsqueeze(1)  # (B, 1, d_model)

        # Attention over node embeddings
        attn_scores = (query @ node_embeddings.transpose(1, 2)) / math.sqrt(
            self.d_model
        )  # (B, 1, N)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, N)

        # Context vector
        context = (attn_weights @ node_embeddings).squeeze(1)  # (B, d_model)

        # Score each node against context
        scores = self.scores_proj(context)  # (B, d_model)
        logits = (node_embeddings @ scores.unsqueeze(-1)).squeeze(-1) / math.sqrt(
            self.d_model
        )  # (B, N)

        logits = logits.masked_fill(mask, float("-inf"))
        return logits  # raw logits, not log_softmax — Categorical handles this