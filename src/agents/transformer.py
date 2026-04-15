from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from src.vrp import VRPEnvironmentBatch


class TransformerEncoder(nn.Module):
    """Transformer encoder over node features."""

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
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(node_features)
        return self.layers(x)


class TransformerDecoder(nn.Module):
    """Two-stage attention decoder that outputs logits for all nodes."""

    CLIP_C = 10.0

    def __init__(self, state_features: int, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

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
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = node_embeddings.shape[0]
        query = self.query_proj(truck_state)

        q = self._split_heads(self.context_q(query.unsqueeze(1)))
        k = self._split_heads(self.context_k(node_embeddings))
        v = self._split_heads(self.context_v(node_embeddings))

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(invalid_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = torch.softmax(attn_scores, dim=-1)

        context = (attn @ v).squeeze(2).reshape(batch_size, self.d_model)
        context = self.context_out(context)

        gq = self.glimpse_q(context)
        gk = self.glimpse_k(node_embeddings)

        logits = (gk @ gq.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.d_model)
        # tanh clipping follows Kool et al.; it prevents early extreme logits.
        logits = self.CLIP_C * torch.tanh(logits)
        logits = logits.masked_fill(invalid_mask, float("-inf"))
        return logits

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, seq, _ = x.shape
        return x.reshape(b, seq, self.num_heads, self.head_dim).transpose(1, 2)


class TransformerAgent:
    """Attention-based policy for VRP batched environments (inference-only here)."""

    def __init__(
        self,
        node_features: int,
        state_features: int,
        d_model: int,
        device: torch.device,
    ):
        self.device = device
        self.encoder = TransformerEncoder(node_features=node_features, d_model=d_model).to(device)
        self.decoder = TransformerDecoder(state_features=state_features, d_model=d_model).to(device)

    def eval(self) -> None:
        self.encoder.eval()
        self.decoder.eval()

    def train(self) -> None:
        self.encoder.train()
        self.decoder.train()

    def forward(
        self,
        node_features: torch.Tensor,
        truck_state: torch.Tensor,
        invalid_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encoder(node_features)
        return self.decoder(embeddings, truck_state, invalid_mask)

    @torch.no_grad()
    def select_actions(self, env: VRPEnvironmentBatch, greedy: bool = True) -> torch.Tensor:
        obs = env.get_observation()
        logits = self.forward(
            node_features=obs["node_features"],
            truck_state=obs["truck_state"],
            invalid_mask=obs["invalid_action_mask"],
        )
        if greedy:
            return logits.argmax(dim=-1).to(torch.long)

        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().to(torch.long)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "node_features": self.encoder.input_proj.in_features,
                "state_features": self.decoder.query_proj.in_features,
                "d_model": self.encoder.input_proj.out_features,
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
            },
            output,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "TransformerAgent":
        ckpt = torch.load(path, map_location=device)
        agent = cls(
            node_features=int(ckpt["node_features"]),
            state_features=int(ckpt["state_features"]),
            d_model=int(ckpt["d_model"]),
            device=device,
        )
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        return agent
