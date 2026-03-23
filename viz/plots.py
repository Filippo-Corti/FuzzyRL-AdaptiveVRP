import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

import config
from env.snapshot import SimulationSnapshot

WINDOW = 500  # Rolling window size for smoothing


class MetricsPlotter:

    def __init__(self):
        self.steps: list[int] = []
        self.rewards: list[float] = []
        self.orphans: list[int] = []
        self.distances: list[float] = []
        self.best_distances: list[float] = []
        self.epsilons: list[float] = []
        self.q_table_sizes: list[int] = []
        self.action_counts: dict[str, list[int]] = {}

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("VRP Fuzzy Q-Learning — Training Metrics", fontsize=13)
        gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.45, wspace=0.3)

        self.ax_reward = self.fig.add_subplot(gs[0, 0])
        self.ax_orphans = self.fig.add_subplot(gs[0, 1])
        self.ax_distance = self.fig.add_subplot(gs[1, 0])
        self.ax_epsilon = self.fig.add_subplot(gs[1, 1])
        self.ax_actions = self.fig.add_subplot(gs[2, 0])
        self.ax_qtable = self.fig.add_subplot(gs[2, 1])

        self._configure_axes()
        plt.ion()
        plt.show()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, snapshot: SimulationSnapshot, epsilon: float, q_table_size: int):
        stats = snapshot.stats
        step = stats.round

        self.steps.append(step)
        self.rewards.append(stats.episode_reward)
        self.orphans.append(stats.orphans)
        self.distances.append(stats.last_distance)
        self.best_distances.append(stats.best_solution_distance)
        self.epsilons.append(epsilon)
        self.q_table_sizes.append(q_table_size)

        action = snapshot.agent_state.chosen_action
        if action not in self.action_counts:
            self.action_counts[action] = [0] * (len(self.steps) - 1)
        for a in self.action_counts:
            self.action_counts[a].append(1 if a == action else 0)

    def draw(self):
        if len(self.steps) < 2:
            return

        self._draw_reward()
        self._draw_orphans()
        self._draw_distance()
        self._draw_epsilon()
        self._draw_actions()
        self._draw_qtable()

        plt.pause(0.001)

    # ------------------------------------------------------------------
    # Private draw methods
    # ------------------------------------------------------------------

    def _draw_reward(self):
        ax = self.ax_reward
        ax.cla()
        ax.set_title("Reward per step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.plot(self.steps, self.rewards, alpha=0.2, color="steelblue", linewidth=0.8)
        ax.plot(
            self.steps,
            _rolling_mean(self.rewards, WINDOW),
            color="steelblue",
            linewidth=1.5,
            label=f"MA-{WINDOW}",
        )
        ax.legend(fontsize=8)

    def _draw_orphans(self):
        ax = self.ax_orphans
        ax.cla()
        ax.set_title("Orphan nodes")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.plot(self.steps, self.orphans, alpha=0.2, color="tomato", linewidth=0.8)
        ax.plot(
            self.steps,
            _rolling_mean(self.orphans, WINDOW),
            color="tomato",
            linewidth=1.5,
            label=f"MA-{WINDOW}",
        )
        ax.legend(fontsize=8)

    def _draw_distance(self):
        ax = self.ax_distance
        ax.cla()
        ax.set_title("Route distance (fully assigned only)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Distance")
        valid_steps = [s for s, d in zip(self.steps, self.distances) if d > 0]
        valid_dists = [d for d in self.distances if d > 0]
        if valid_steps:
            ax.scatter(valid_steps, valid_dists, s=4, alpha=0.3, color="mediumseagreen")
        ax.plot(
            self.steps,
            self.best_distances,
            color="darkgreen",
            linewidth=1.5,
            label="Best so far",
        )
        ax.legend(fontsize=8)

    def _draw_epsilon(self):
        ax = self.ax_epsilon
        ax.cla()
        ax.set_title("Epsilon (exploration rate)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Epsilon")
        ax.plot(self.steps, self.epsilons, color="darkorchid", linewidth=1.5)
        ax.set_ylim(0, 1)

    def _draw_actions(self):
        ax = self.ax_actions
        ax.cla()
        ax.set_title(f"Action distribution (MA-{WINDOW})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Frequency")
        colors = ["steelblue", "tomato", "mediumseagreen", "darkorchid"]
        for (action, counts), color in zip(self.action_counts.items(), colors):
            smoothed = _rolling_mean(counts, WINDOW)
            ax.plot(self.steps, smoothed, label=action, color=color, linewidth=1.5)
        ax.legend(fontsize=7)

    def _draw_qtable(self):
        ax = self.ax_qtable
        ax.cla()
        ax.set_title("Q-table coverage")
        ax.set_xlabel("Step")
        ax.set_ylabel("Non-zero entries")
        ax.plot(self.steps, self.q_table_sizes, color="sienna", linewidth=1.5)

    def _configure_axes(self):
        for ax in [
            self.ax_reward,
            self.ax_orphans,
            self.ax_distance,
            self.ax_epsilon,
            self.ax_actions,
            self.ax_qtable,
        ]:
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _rolling_mean(values: list[float], window: int) -> list[float]:
    result = []
    q: deque = deque(maxlen=window)
    for v in values:
        q.append(v)
        result.append(sum(q) / len(q))
    return result
