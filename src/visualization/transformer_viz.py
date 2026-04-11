from __future__ import annotations

import torch

from ..agent.transformer_agent import TransformerAgent
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
        **kwargs: object,
    ) -> "TransformerVisualization":
        d_model_obj = kwargs.get("d_model", 128)
        speed_obj = kwargs.get("speed", 0.05)
        seed_obj = kwargs.get("seed", 42)

        assert isinstance(d_model_obj, int)
        assert isinstance(speed_obj, (int, float))
        assert isinstance(seed_obj, int)

        d_model = d_model_obj
        speed = float(speed_obj)
        seed = seed_obj
        device = kwargs.get("device")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            assert isinstance(device, torch.device)

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

    def reload_checkpoint(self, checkpoint_path: str, device: torch.device) -> int:
        agent = self.agent
        if not isinstance(agent, TransformerAgent):
            raise TypeError("TransformerVisualization requires a TransformerAgent")
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.eval()
        return int(ckpt.get("episode", 0))
