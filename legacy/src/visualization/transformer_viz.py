from __future__ import annotations

import torch
import config

from ..agent.base import AgentObservation
from ..agent.transformer.agent import TransformerAgent
from .base import BaseVisualization


class TransformerVisualization(BaseVisualization):
    """
    Visualization for the TransformerAgent.
    All animation logic is in BaseVisualization — this class only implements
    _select_action() and load_agent().
    """

    def _select_action(self) -> torch.Tensor:
        with torch.no_grad():
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
    ) -> "TransformerVisualization":
        d_model_obj = kwargs.get("d_model", config.TRANSFORMER_D_MODEL)
        speed_obj = kwargs.get("speed", 0.05)
        seed_obj = kwargs.get("seed", 42)

        assert isinstance(d_model_obj, int)
        assert isinstance(speed_obj, (int, float))
        assert isinstance(seed_obj, (int, type(None)))

        d_model = d_model_obj
        speed = float(speed_obj)
        seed = seed_obj
        device = kwargs.get("device")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            assert isinstance(device, torch.device)

        agent = TransformerAgent(
            node_features=config.TRANSFORMER_NODE_FEATURES,
            state_features=config.TRANSFORMER_STATE_FEATURES,
            d_model=d_model,
            device=device,
            optimizer_lr=config.TRANSFORMER_LR,
        )
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.eval()

        return cls(
            agent=agent, num_nodes=num_nodes, device=device, speed=speed, seed=seed
        )

    def _reload_agent_state(
        self,
        checkpoint_path: str,
        device: torch.device | None = None,
    ) -> float:
        agent = self.agent
        if not isinstance(agent, TransformerAgent):
            raise TypeError("TransformerVisualization requires a TransformerAgent")
        map_device = device if device is not None else self.device
        ckpt = torch.load(checkpoint_path, map_location=map_device)
        agent.encoder.load_state_dict(ckpt["encoder"])
        agent.decoder.load_state_dict(ckpt["decoder"])
        agent.eval()
        return float(ckpt.get("episode", 0))
