import torch
import torch.nn as nn
import math


class Decoder(nn.Module):
    """
    Builds a query from the current state (input_dim) and uses attention to compute a distribution over the next node to visit.
    """

    def __init__(self, state_features: int, d_model: int):
        super(Decoder, self).__init__()
        self.input_dim = state_features
        self.d_model = d_model
        self.query_proj = nn.Linear(self.input_dim, self.d_model)
        self.scores_proj = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        node_embeddings: torch.Tensor,  # (batch, N, d_model)
        query_state: torch.Tensor,  # (batch, input_dim)
        mask: torch.Tensor,  # (batch, N)
    ) -> torch.Tensor:
        # 1) Build attention scores, using node_embeddings and the query
        query = self.query_proj(query_state)  # (batch, d_model)
        query_expanded = query.unsqueeze(
            1
        )  # (batch, 1, d_model) <- Required for batch matrix multiplication

        attn_scores = (query_expanded @ node_embeddings.transpose(1, 2)) / math.sqrt(
            self.d_model
        )  # (batch, 1, N)

        attn_scores = attn_scores.masked_fill(
            mask.unsqueeze(1), float("-inf")
        )  # Mask out invalid nodes

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, 1, N)
        context = attn_weights @ node_embeddings  # (batch, 1, d_model)
        context = context.squeeze(1)  # (batch, d_model)

        scores = self.scores_proj(context)

        # 2) Build a distribution over the next node to visit, using the attention scores
        logits = (node_embeddings @ scores.unsqueeze(-1)).squeeze(-1) / math.sqrt(
            self.d_model
        )  # (batch, N, 1)

        logits = logits.masked_fill(mask, float("-inf"))
        return torch.log_softmax(logits, dim=-1)
