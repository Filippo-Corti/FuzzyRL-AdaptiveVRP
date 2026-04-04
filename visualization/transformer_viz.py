from __future__ import annotations

import torch

from agent.transformer_agent import TransformerAgent
from .base import BaseVisualization


class TransformerVisualization(BaseVisualization):
    """
    Visualization for the TransformerAgent.
    All animation logic is in BaseVisualization — this class only implements
    _select_action() and load_agent().
    """

    def _select_action(self) -> torch.Tensor:
        with torch.no_grad():
            node_features, truck_state, mask = self.env.get_state()
            actions, _ = self.agent.select_action(
                node_features, truck_state, mask, greedy=True
            )
        return actions  # (1,)

    @classmethod
    def load_agent(
        cls,
        checkpoint_path: str,
        num_nodes: int,
        d_model: int = 128,
        speed: float = 0.05,
        seed: int = 42,
        device: torch.device | None = None,
    ) -> "TransformerVisualization":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = TransformerAgent(
            node_features=4, state_features=3, d_model=d_model, device=device
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.eval()

        return cls(
            agent=agent, num_nodes=num_nodes, device=device, speed=speed, seed=seed
        )
