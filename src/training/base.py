from __future__ import annotations

from abc import ABC, abstractmethod


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
        ...

    @abstractmethod
    def step(self) -> dict:
        """
        Execute one environment step.
        Returns a metrics dict (contents are subclass-defined).
        Should only be called after reset_episode() and while not is_done().
        """
        ...

    @abstractmethod
    def update(self) -> float:
        """
        Apply the learning update for the completed episode.
        Should only be called after is_done() is True.
        Returns a scalar loss or TD-error for logging.
        """
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """True when the current episode is complete."""
        ...

    @abstractmethod
    def train(self, num_episodes: int) -> None:
        """Run the full training loop."""
        ...

    @abstractmethod
    def save(self, path: str | None = None) -> None:
        """Persist trainer + agent state to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str, **kwargs) -> "BaseTrainer":
        """Restore trainer + agent state from disk."""
        ...
