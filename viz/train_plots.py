# training/visualize_training.py
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

from simulation.breakdown_training import EpisodeResult

WINDOW = 50


class BreakdownTrainingPlotter:

    def __init__(self):
        self.results: list[EpisodeResult] = []

        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("Breakdown agent — training metrics", fontsize=13)
        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.5, wspace=0.35)

        self.ax_steps = self.fig.add_subplot(gs[0, 0])
        self.ax_reward = self.fig.add_subplot(gs[0, 1])
        self.ax_orphans = self.fig.add_subplot(gs[1, 0])
        self.ax_dist = self.fig.add_subplot(gs[1, 1])
        self.ax_epsilon = self.fig.add_subplot(gs[2, 0])
        self.ax_qtable = self.fig.add_subplot(gs[2, 1])

        for ax in self.axes:
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    @property
    def axes(self):
        return [
            self.ax_steps,
            self.ax_reward,
            self.ax_orphans,
            self.ax_dist,
            self.ax_epsilon,
            self.ax_qtable,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, result: EpisodeResult):
        self.results.append(result)

    def draw(self):
        if len(self.results) < 2:
            return
        for ax in self.axes:
            ax.cla()
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)

        episodes = list(range(1, len(self.results) + 1))
        self._draw_steps(episodes)
        self._draw_reward(episodes)
        self._draw_orphans(episodes)
        self._draw_dist(episodes)
        self._draw_epsilon(episodes)
        self._draw_qtable(episodes)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def show(self):
        """Call after training completes for a final static view."""
        self.draw()
        plt.ioff()
        plt.show()

    def save(self, path: str = "training_metrics.png"):
        self.draw()
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved to {path}")

    # ------------------------------------------------------------------
    # Panel draw methods
    # ------------------------------------------------------------------

    def _draw_steps(self, episodes: list[int]):
        ax = self.ax_steps
        ax.set_title("Steps per episode", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Steps", fontsize=8)

        steps = [r.steps for r in self.results]
        colors = ["#2ecc71" if r.success else "#e74c3c" for r in self.results]
        ax.bar(episodes, steps, color=colors, alpha=0.5, width=1.0)
        ax.plot(
            episodes,
            _rolling_mean(steps, WINDOW),
            color="#e67e22",
            linewidth=1.5,
            label=f"MA-{WINDOW}",
        )

        # Success/timeout legend
        ax.bar([], [], color="#2ecc71", alpha=0.6, label="Success")
        ax.bar([], [], color="#e74c3c", alpha=0.6, label="Timeout")
        ax.axhline(100, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

    def _draw_reward(self, episodes: list[int]):
        ax = self.ax_reward
        ax.set_title("Total reward per episode", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Reward", fontsize=8)

        rewards = [r.total_reward for r in self.results]
        ax.plot(episodes, rewards, alpha=0.25, color="#3498db", linewidth=0.8)
        ax.plot(
            episodes,
            _rolling_mean(rewards, WINDOW),
            color="#3498db",
            linewidth=1.5,
            label=f"MA-{WINDOW}",
        )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7)

    def _draw_orphans(self, episodes: list[int]):
        ax = self.ax_orphans
        ax.set_title("Orphans remaining on timeout", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Orphan count", fontsize=8)

        timeout_eps = [e for e, r in zip(episodes, self.results) if not r.success]
        timeout_orphans = [r.final_orphans for r in self.results if not r.success]

        if timeout_eps:
            ax.scatter(
                timeout_eps,
                timeout_orphans,
                color="#e74c3c",
                s=6,
                alpha=0.5,
                label="Timeout episodes",
            )
            ax.plot(
                timeout_eps,
                _rolling_mean(timeout_orphans, WINDOW),
                color="#e74c3c",
                linewidth=1.5,
                label=f"MA-{WINDOW}",
            )
        ax.legend(fontsize=7)

    def _draw_dist(self, episodes: list[int]):
        ax = self.ax_dist
        ax.set_title("Final distance — successful episodes", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Total distance", fontsize=8)

        success_eps = [e for e, r in zip(episodes, self.results) if r.success]
        success_dists = [r.final_distance for r in self.results if r.success]

        if success_eps:
            ax.scatter(success_eps, success_dists, color="#9b59b6", s=6, alpha=0.5)
            ax.plot(
                success_eps,
                _rolling_mean(success_dists, WINDOW),
                color="#9b59b6",
                linewidth=1.5,
                label=f"MA-{WINDOW}",
            )
        ax.legend(fontsize=7)

    def _draw_epsilon(self, episodes: list[int]):
        ax = self.ax_epsilon
        ax.set_title("Epsilon decay", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Epsilon", fontsize=8)
        ax.set_ylim(0, 1)

        epsilons = [r.epsilon for r in self.results]
        ax.plot(episodes, epsilons, color="#f39c12", linewidth=1.5)

    def _draw_qtable(self, episodes: list[int]):
        ax = self.ax_qtable
        ax.set_title("Q-table size", fontsize=9)
        ax.set_xlabel("Episode", fontsize=8)
        ax.set_ylabel("Entries", fontsize=8)

        sizes = [r.q_table_size for r in self.results]
        ax.plot(episodes, sizes, color="#7f8c8d", linewidth=1.5)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result = []
    q: deque = deque(maxlen=window)
    for v in values:
        q.append(v)
        result.append(sum(q) / len(q))
    return result
