from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


@dataclass
class AgentObservation:
    """Semantic wrapper for the tensors consumed by policies."""

    node_features: torch.Tensor  # (B, N+1, 4)
    truck_state: torch.Tensor  # (B, 3)
    mask: torch.Tensor  # (B, N+1) bool


@dataclass
class AgentDecision:
    """Semantic wrapper for selected actions and their log probabilities."""

    actions: torch.Tensor  # (B,) int
    log_probs: torch.Tensor  # (B,) float


class BaseAgent(ABC):
    """
    Minimal interface that all VRP agents must implement.

    The only method the training and visualization infrastructure depends on
    is select_action — everything else (save, load) is for persistence.
    log_probs is meaningful for gradient-based agents and zeros for Q-table
    agents; callers that don't need it (viz, fuzzy trainer) simply ignore it.
    """

    @abstractmethod
    def select_action(
        self,
        observation: AgentObservation,
        greedy: bool = False,
    ) -> AgentDecision:
        """
        Select actions for a batch of instances.

        Returns:
            AgentDecision with:
                actions:   (B,) int   — index into node_features (0=depot, 1..N=customer)
                log_probs: (B,) float — log probability of each action.
                                        Return zeros if not applicable.
        """
        pass

    def eval(self) -> None:
        """Optional hook for agents with train/eval mode."""

    def train(self) -> None:
        """Optional hook for agents with train/eval mode."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: torch.device) -> "BaseAgent":
        """Restore agent state from disk."""
        pass
