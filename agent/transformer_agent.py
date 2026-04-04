from __future__ import annotations

import itertools
import torch

from .base import BaseAgent
from .encoder import Encoder
from .decoder import Decoder
from simulation.snapshot import AgentSnapshot


class TransformerAgent(BaseAgent):
    """
    Attention-based policy for VRP construction.

    During batched training, call forward() directly.
    During single-instance visualisation, uses VRPAgent interface.
    """

    def __init__(
        self,
        node_features: int,
        state_features: int,
        d_model: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.encoder = Encoder(node_features, d_model).to(device)
        self.decoder = Decoder(state_features, d_model).to(device)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=1e-4,
        )

        # Single-instance viz state
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, 4)
        truck_state: torch.Tensor,  # (B, 3)
        mask: torch.Tensor,  # (B, N) bool
    ) -> torch.Tensor:
        """Returns raw logits (B, N). Used directly by training loop."""
        embeddings = self.encoder(node_features)
        return self.decoder(embeddings, truck_state, mask)

    def select_action(
        self,
        node_features: torch.Tensor,
        truck_state: torch.Tensor,
        mask: torch.Tensor,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Batched action selection. Used by training loop.

        returns:
            actions: (B,) int
            log_probs: (B,) float
        """
        logits = self.forward(node_features, truck_state, mask)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str, device: torch.device) -> "TransformerAgent":
        pass

    def finish_episode(self, baseline: float | None = None):
        """Used only during single-instance viz mode."""
        self._log_probs = []
        self._rewards = []

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            last_choice=None,  # TODO?
            epsilon=None,
        )
