from __future__ import annotations

import itertools

import torch

from ..base import AgentDecision, AgentObservation, BaseAgent
from .decoder import Decoder
from .encoder import Encoder


class TransformerAgent(BaseAgent):
	"""Attention-based policy for VRP construction."""

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
		self._log_probs: list[torch.Tensor] = []
		self._rewards: list[float] = []

	def forward(
		self,
		node_features: torch.Tensor,
		truck_state: torch.Tensor,
		mask: torch.Tensor,
	) -> torch.Tensor:
		embeddings = self.encoder(node_features)
		return self.decoder(embeddings, truck_state, mask)

	def select_action(
		self,
		observation: AgentObservation,
		greedy: bool = False,
	) -> AgentDecision:
		logits = self.forward(
			observation.node_features,
			observation.truck_state,
			observation.mask,
		)
		dist = torch.distributions.Categorical(logits=logits)
		if greedy:
			actions = logits.argmax(dim=-1)
		else:
			actions = dist.sample()
		log_probs = dist.log_prob(actions)
		return AgentDecision(actions=actions, log_probs=log_probs)

	def save(self, path: str) -> None:
		torch.save(
			{
				"node_features": self.encoder.input_proj.in_features,
				"state_features": self.decoder.query_proj.in_features,
				"d_model": self.encoder.input_proj.out_features,
				"encoder": self.encoder.state_dict(),
				"decoder": self.decoder.state_dict(),
				"optimizer": self.optimizer.state_dict(),
			},
			path,
		)

	@classmethod
	def load(cls, path: str, device: torch.device) -> "TransformerAgent":
		ckpt = torch.load(path, map_location=device)
		agent = cls(
			node_features=ckpt.get("node_features", 4),
			state_features=ckpt.get("state_features", 3),
			d_model=ckpt.get("d_model", 128),
			device=device,
		)
		agent.encoder.load_state_dict(ckpt["encoder"])
		agent.decoder.load_state_dict(ckpt["decoder"])
		optimizer_state = ckpt.get("optimizer")
		if optimizer_state is not None:
			agent.optimizer.load_state_dict(optimizer_state)
		return agent

	def finish_episode(self, baseline: float | None = None):
		self._log_probs = []
		self._rewards = []

	def eval(self):
		self.encoder.eval()
		self.decoder.eval()

	def train(self):
		self.encoder.train()
		self.decoder.train()


__all__ = ["TransformerAgent"]
