from __future__ import annotations

import torch

from ..agent.fuzzy_agent import FuzzyAgent
from .base import BaseVisualization


class FuzzyVisualization(BaseVisualization):
    """
    Visualization for the FuzzyAgent.
    All animation logic is in BaseVisualization — this class only implements
    _select_action() and load_agent().
    """

    def _select_action(self) -> torch.Tensor:
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
    ) -> "FuzzyVisualization":
        speed_obj = kwargs.get("speed", 0.05)
        seed_obj = kwargs.get("seed", 42)

        assert isinstance(speed_obj, (int, float))
        assert isinstance(seed_obj, int)

        speed = float(speed_obj)
        seed = seed_obj
        device = kwargs.get("device")
        if device is None:
            device = torch.device("cpu")
        else:
            assert isinstance(device, torch.device)
        agent = FuzzyAgent.load(checkpoint_path, device=device)
        return cls(
            agent=agent, num_nodes=num_nodes, device=device, speed=speed, seed=seed
        )

    def reload_checkpoint(self, checkpoint_path: str) -> float:
        self.agent = FuzzyAgent.load(checkpoint_path)
        return float(self.agent.epsilon)
