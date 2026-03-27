from __future__ import annotations

import itertools
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from agent import VRPAgent
from .encoder import Encoder
from .decoder import Decoder
from simulation.snapshot import AgentSnapshot

if TYPE_CHECKING:
    from env import EnvObservation, NodeObservation


class TransformerAgent(VRPAgent):
    """
    Attention-based policy for VRP construction.

    During batched training, call forward() directly.
    During single-instance visualisation, uses VRPAgent interface.
    """

    def __init__(self, node_features: int, state_features: int, d_model: int):
        super().__init__()
        self.encoder = Encoder(node_features, d_model)
        self.decoder = Decoder(state_features, d_model)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=1e-4,
        )

        # Single-instance viz state
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self.last_node: NodeObservation | None = None

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, 4)
        truck_state: torch.Tensor,     # (B, 3)
        mask: torch.Tensor,            # (B, N) bool
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

    # --- VRPAgent interface for single-instance visualisation ---

    def select_node(self, obs: EnvObservation, greedy: bool = False) -> int:
        from training.batch_env import BatchVRPEnv
        node_features, truck_state, mask = self._obs_to_tensors(obs)
        logits = self.forward(node_features, truck_state, mask)
        logits = logits.squeeze(0)

        if greedy:
            idx = logits.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            idx = dist.sample().item()
            self._log_probs.append(dist.log_prob(torch.tensor(idx)))

        self.last_node = obs.nodes[idx]
        return self.last_node.id

    def update(self, reward: float, obs: EnvObservation):
        self._rewards.append(reward)

    def finish_episode(self, baseline: float | None = None):
        """Used only during single-instance viz mode."""
        self._log_probs = []
        self._rewards = []

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            last_choice=(
                (self.last_node.x, self.last_node.y) if self.last_node else None
            ),
            epsilon=None,
        )

    def record(self, obs: EnvObservation):
        pass

    def _obs_to_tensors(self, obs: EnvObservation):
        remaining = obs.truck_capacity - obs.truck_load
        node_features, mask = [], []
        for node in obs.nodes:
            node_features.append([
                node.x, node.y,
                node.demand / obs.truck_capacity,
                float(node.depot),
            ])
            if node.depot:
                mask.append(obs.truck_at_depot)
            else:
                mask.append(node.visited or node.demand > remaining)

        nf = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0)
        ts = torch.tensor([
            obs.truck_pos[0], obs.truck_pos[1], remaining / obs.truck_capacity
        ], dtype=torch.float32).unsqueeze(0)
        mk = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        return nf, ts, mk