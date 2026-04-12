from __future__ import annotations

import torch

from ..agent.base import AgentObservation
from ..agent.fuzzy.agent import FuzzyAgent
from .base import BaseVisualization


class FuzzyVisualization(BaseVisualization):
    """
    Visualization for the FuzzyAgent.
    All animation logic is in BaseVisualization — this class only implements
    _select_action() and load_agent().
    """

    def _select_action(self) -> torch.Tensor:
        env_obs = self.env.get_state()
        decision = self.agent.select_action(
            AgentObservation(
                node_features=env_obs.node_features,
                truck_state=env_obs.truck_state,
                mask=env_obs.mask,
            ),
            greedy=True,
        )
        return decision.actions  # (1,)

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
        assert isinstance(seed_obj, (int, type(None)))

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

    def _reload_agent_state(
        self,
        checkpoint_path: str,
        device: torch.device | None = None,
    ) -> float:
        load_device = device if device is not None else self.device
        self.agent = FuzzyAgent.load(checkpoint_path, device=load_device)
        return float(self.agent.epsilon)
