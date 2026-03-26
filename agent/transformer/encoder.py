import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Applies a transformation that projects node features (input_dim) into an embedding space (d_model).
    """

    def __init__(self, nodes_features: int, d_model: int):
        super(Encoder, self).__init__()
        self.input_dim = nodes_features
        self.d_model = d_model
        self.proj = nn.Linear(self.input_dim, self.d_model)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.proj(nodes)
