from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..base import AgentDecision, AgentObservation, BaseAgent


def _triangular(x: float, a: float, b: float, c: float) -> float:
	"""Triangular membership function with peak at b, zero at a and c."""
	if x <= a or x >= c:
		return 0.0
	if x <= b:
		return (x - a) / (b - a) if b > a else 1.0
	return (c - x) / (c - b) if c > b else 1.0


def fuzzify(
	value: float, breakpoints: list[tuple[str, float, float, float]]
) -> dict[str, float]:
	"""
	Fuzzify a crisp value using a list of (label, a, b, c) triangular MFs.
	Returns {label: membership} for all non-zero memberships.
	"""
	result = {}
	for label, a, b, c in breakpoints:
		mu = _triangular(value, a, b, c)
		if mu > 0.0:
			result[label] = mu
	return result


BREAKPOINTS = {
	"dist_nearest": [
		("L", 0.0, 0.0, 0.3),
		("M", 0.1, 0.3, 0.6),
		("H", 0.4, 1.0, 1.0),
	],
	"demand_nearest": [
		("L", 0.0, 0.0, 0.4),
		("M", 0.2, 0.5, 0.8),
		("H", 0.6, 1.0, 1.0),
	],
	"dist_2nd": [
		("L", 0.0, 0.0, 0.3),
		("M", 0.1, 0.3, 0.6),
		("H", 0.4, 1.0, 1.0),
	],
	"demand_2nd": [
		("L", 0.0, 0.0, 0.4),
		("M", 0.2, 0.5, 0.8),
		("H", 0.6, 1.0, 1.0),
	],
	"dist_3rd": [
		("L", 0.0, 0.0, 0.3),
		("M", 0.1, 0.3, 0.6),
		("H", 0.4, 1.0, 1.0),
	],
	"demand_3rd": [
		("L", 0.0, 0.0, 0.4),
		("M", 0.2, 0.5, 0.8),
		("H", 0.6, 1.0, 1.0),
	],
	"remaining_cap": [
		("L", 0.0, 0.0, 0.4),
		("M", 0.2, 0.5, 0.8),
		("H", 0.6, 1.0, 1.0),
	],
}

SIGNAL_NAMES = list(BREAKPOINTS.keys())
N_ACTIONS = 4


class FuzzyAgent(BaseAgent):
	"""Fuzzy Q-learning policy for VRP action selection."""

	def __init__(
		self,
		epsilon: float = 0.9,
		epsilon_min: float = 0.05,
		epsilon_decay: float = 0.995,
		lr: float = 0.1,
		gamma: float = 0.95,
	):
		self.epsilon = epsilon
		self.epsilon_min = epsilon_min
		self.epsilon_decay = epsilon_decay
		self.lr = lr
		self.gamma = gamma
		self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * N_ACTIONS)
		self._last_state_weights: list[tuple[tuple, float]] | None = None
		self._last_action: int | None = None

	def select_action(
		self,
		observation: AgentObservation,
		greedy: bool = False,
	) -> AgentDecision:
		signals, candidates = self._extract_signals(
			observation.node_features,
			observation.truck_state,
			observation.mask,
		)
		if signals is None:
			self._last_state_weights = None
			self._last_action = 0
			return AgentDecision(actions=torch.tensor([0]), log_probs=torch.zeros(1))

		assert candidates is not None
		state_weights = self._get_state_weights(signals)
		q_values = self._aggregate_q(state_weights)

		valid_actions = list(range(len(candidates))) + [N_ACTIONS - 1]
		if observation.mask[0, 0].item():
			valid_actions = [a for a in valid_actions if a != N_ACTIONS - 1]
		if not valid_actions:
			valid_actions = [N_ACTIONS - 1]

		if not greedy and np.random.random() < self.epsilon:
			fuzzy_action = np.random.choice(valid_actions)
		else:
			fuzzy_action = max(valid_actions, key=lambda a: q_values[a])

		self._last_state_weights = state_weights
		self._last_action = fuzzy_action
		env_action = self._fuzzy_action_to_env(
			fuzzy_action, candidates, observation.mask
		)
		return AgentDecision(actions=torch.tensor([env_action]), log_probs=torch.zeros(1))

	def q_update(
		self,
		reward: float,
		next_node_features: torch.Tensor,
		next_truck_state: torch.Tensor,
		next_mask: torch.Tensor,
		done: bool,
	):
		if self._last_state_weights is None:
			return
		assert self._last_action is not None

		if done:
			target = reward
		else:
			next_signals, _ = self._extract_signals(
				next_node_features,
				next_truck_state,
				next_mask,
			)
			if next_signals is None:
				target = reward
			else:
				next_weights = self._get_state_weights(next_signals)
				next_q = self._aggregate_q(next_weights)
				target = reward + self.gamma * max(next_q)

		for state_key, weight in self._last_state_weights:
			current_q = self.q_table[state_key][self._last_action]
			self.q_table[state_key][self._last_action] += (
				self.lr * weight * (target - current_q)
			)

	def decay_epsilon(self):
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

	@property
	def last_fuzzy_action(self) -> int | None:
		return self._last_action

	def save(self, path: str) -> None:
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		with open(path, "wb") as f:
			pickle.dump(
				{
					"q_table": dict(self.q_table),
					"epsilon": self.epsilon,
				},
				f,
			)

	@classmethod
	def load(cls, path: str, device: torch.device | None = None) -> "FuzzyAgent":
		with open(path, "rb") as f:
			data = pickle.load(f)
		agent = cls()
		agent.q_table = defaultdict(lambda: [0.0] * N_ACTIONS, data["q_table"])
		agent.epsilon = data["epsilon"]
		return agent

	def _extract_signals(
		self,
		node_features: torch.Tensor,
		truck_state: torch.Tensor,
		mask: torch.Tensor,
	) -> tuple[dict[str, float] | None, list[int] | None]:
		nf = node_features[0]
		ts = truck_state[0]
		mk = mask[0]

		truck_xy = ts[:2]
		remaining_cap = ts[2].item()

		customer_xy = nf[1:, :2]
		customer_valid = ~mk[1:]

		valid_indices = torch.where(customer_valid)[0]
		if len(valid_indices) == 0:
			return None, None

		diffs = customer_xy[valid_indices] - truck_xy.unsqueeze(0)
		dists = torch.norm(diffs, dim=-1)
		sorted_order = torch.argsort(dists)
		top_k = sorted_order[:3]
		top_indices = valid_indices[top_k]
		top_dists = dists[top_k]
		top_demands = nf[1:, 2][top_indices]

		pad = 3 - len(top_indices)
		if pad > 0:
			top_dists = torch.cat([top_dists, torch.ones(pad)])
			top_demands = torch.cat([top_demands, torch.zeros(pad)])

		signals = {
			"dist_nearest": top_dists[0].item(),
			"demand_nearest": top_demands[0].item(),
			"dist_2nd": top_dists[1].item(),
			"demand_2nd": top_demands[1].item(),
			"dist_3rd": top_dists[2].item(),
			"demand_3rd": top_demands[2].item(),
			"remaining_cap": remaining_cap,
		}
		candidates = [int(idx.item()) + 1 for idx in top_indices]
		return signals, candidates

	def _get_state_weights(self, signals: dict) -> list[tuple[tuple, float]]:
		per_signal = []
		for name in SIGNAL_NAMES:
			memberships = fuzzify(signals[name], BREAKPOINTS[name])
			if not memberships:
				memberships = {"H": 1.0} if signals[name] >= 1.0 else {"L": 1.0}
			per_signal.append(memberships)

		state_weights: list[tuple[dict[int, str], float]] = [({}, 1.0)]
		for memberships in per_signal:
			new_state_weights = []
			for partial_key, partial_weight in state_weights:
				for label, mu in memberships.items():
					new_key = {**partial_key, len(partial_key): label}
					new_state_weights.append((new_key, partial_weight * mu))
			state_weights = new_state_weights

		return [(tuple(d.values()), w) for d, w in state_weights]

	def _aggregate_q(self, state_weights: list[tuple[tuple, float]]) -> list[float]:
		total_weight = sum(w for _, w in state_weights)
		q_values = [0.0] * N_ACTIONS
		for state_key, weight in state_weights:
			for a in range(N_ACTIONS):
				q_values[a] += (weight / total_weight) * self.q_table[state_key][a]
		return q_values

	def _fuzzy_action_to_env(
		self, fuzzy_action: int, candidates: list[int], mask: torch.Tensor
	) -> int:
		if fuzzy_action == N_ACTIONS - 1:
			return 0
		if fuzzy_action < len(candidates):
			return candidates[fuzzy_action]
		return 0


__all__ = ["FuzzyAgent"]
