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

    def __init__(self, nodes_features: int, state_features: int, d_model: int):
        super().__init__()
        self.encoder = Encoder(nodes_features, d_model)
        self.decoder = Decoder(state_features, d_model)
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=1e-4,
        )

        self.log_probs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.last_node: NodeObservation | None = None

    def select_node(self, obs: EnvObservation, greedy: bool = False) -> int:
        log_probs, node_ids = self.forward(obs)
        log_probs = log_probs.squeeze(0)
        if greedy:
            idx = torch.argmax(log_probs, dim=-1).item()
        else:
            idx = torch.distributions.Categorical(logits=log_probs).sample().item()

        self.last_node = obs.nodes[idx]
        return self.last_node.id

    def record(self, obs: EnvObservation):
        """
        Explicitly record log prob of the last action taken. Call during training only.
        """
        log_probs, node_ids = self.forward(obs)
        log_probs = log_probs.squeeze(0)
        dist = torch.distributions.Categorical(logits=log_probs)
        idx = node_ids.index(self.last_node.id)
        self.log_probs.append(dist.log_prob(torch.tensor(idx)))

    def update(self, reward: float, obs: EnvObservation):
        self.rewards.append(reward)

    def finish_episode(self, baseline: float | None = None):
        if not self.log_probs:
            return

        R = sum(self.rewards)
        if baseline is None:
            baseline = self.compute_baseline()

        self.optimizer.zero_grad()
        loss = -(R - baseline) * sum(self.log_probs)
        loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []

    def snapshot(self) -> AgentSnapshot:
        return AgentSnapshot(
            last_choice=(
                (self.last_node.x, self.last_node.y) if self.last_node else None
            ),
            epsilon=None,
        )

    def forward(self, obs: EnvObservation) -> tuple[torch.Tensor, list[int]]:
        nodes_tensor, truck_tensor, mask_tensor = self.obs_to_tensors(obs)
        node_embeddings = self.encoder(nodes_tensor)
        log_probs = self.decoder(node_embeddings, truck_tensor, mask_tensor)
        node_ids = [node.id for node in obs.nodes]
        return log_probs, node_ids

    def obs_to_tensors(
        self, obs: EnvObservation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Converts the observation into tensors, returning the node features, the truck state and the mask for invalid nodes.
        """
        max_demand = obs.truck_capacity - obs.truck_load
        state_features = [
            obs.truck_pos[0],
            obs.truck_pos[1],
            max_demand / obs.truck_capacity,
        ]
        node_features = []
        mask = []  # True means invalid
        for node in obs.nodes:
            is_depot = node.depot
            demand_fraction = node.demand / obs.truck_capacity

            node_features.append([node.x, node.y, demand_fraction, float(is_depot)])
            if is_depot:  # TODO: issue here
                mask.append(False)
            else:
                mask.append(node.visited or node.demand > max_demand)

        node_tensors = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0)
        truck_tensors = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)

        return node_tensors, truck_tensors, mask_tensor

    def compute_baseline(self) -> float:
        return 0.0
