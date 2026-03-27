import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Applies a transformation that projects node features (input_dim) into an embedding space (d_model).
    """

    def __init__(self, nodes_features: int, d_model: int):
        super(Encoder, self).__init__()
        self.proj = nn.Linear(nodes_features, d_model)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.proj(nodes)
