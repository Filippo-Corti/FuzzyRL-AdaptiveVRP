from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class BaseTrainer(ABC):
    """
    Common interface for all VRP trainers.

    Exposes both a step-by-step interface (reset_episode / step / update)
    for external control and a full-loop train() method for convenience.
    Subclasses handle their own learning algorithm internally.
    """

    @abstractmethod
    def reset_episode(self) -> None:
        """Reset environment and any per-episode accumulators."""
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Execute one environment step.
        Should only be called after reset_episode() and while not is_done().
        """
        pass

    @abstractmethod
    def update(self) -> float:
        """
        Apply the learning update for the completed episode.
        Should only be called after is_done() is True.
        Returns a scalar loss or TD-error for logging.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """True when the current episode is complete."""
        pass

    @abstractmethod
    def train(
        self,
        num_episodes: int,
        progress_callback: Callable[[dict[str, int | float | None]], None] | None = None,
    ) -> None:
        """Run the full training loop."""
        pass

    @abstractmethod
    def save(self, path: str | None = None) -> None:
        """Persist trainer + agent state to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "BaseTrainer":
        """Restore trainer + agent state from disk."""
        pass
