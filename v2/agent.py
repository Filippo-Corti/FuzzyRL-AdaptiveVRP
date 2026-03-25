import numpy as np
from typing import Dict, List, Tuple

from .fuzzy import (
    fuzzify,
    build_qtable,
    fuzzy_q_values,
    fuzzy_q_update,
    ALL_LABEL_COMBOS,
)
from environment import VRPEnvironment


class FuzzyQAgent:
    """
    Fuzzy Q-learning agent using Monte-Carlo episode returns.

    Why Monte-Carlo instead of TD(0)?
    ----------------------------------
    The reward is only observed at episode end (total route length).
    MC updates every step in the episode with the same final return,
    which gives a clean unbiased signal: good episodes reinforce all
    the decisions that led to them.
    """

    def __init__(
        self,
        n_actions: int = 4,  # must match env.N_ACTIONS
        alpha: float = 0.25,
        gamma: float = 1.0,  # no discounting — full route quality matters
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9975,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.qtable = build_qtable(n_actions)

        # History for plotting
        self.episode_distances: List[float] = []
        self.episode_returns: List[float] = []

    # -------------------------------------------------------------- Selection

    def select_action(self, obs: dict) -> Tuple[int, Dict]:
        memberships = fuzzify(obs)
        candidates = obs["candidates"]

        # Force return if nothing to visit
        if len(candidates) == 0:
            return self.n_actions - 1, memberships  # ACTION_RETURN

        if np.random.random() < self.epsilon:
            # Explore: uniform over feasible actions (all candidates + return)
            n_feasible = len(candidates) + 1  # +1 for return
            action = np.random.randint(n_feasible)
        else:
            qv = fuzzy_q_values(self.qtable, memberships, self.n_actions)
            # Mask out candidate slots that don't exist
            if len(candidates) < self.n_actions - 1:
                qv[len(candidates) : self.n_actions - 1] = -np.inf
            action = int(np.argmax(qv))

        return action, memberships

    # -------------------------------------------------------------- Episode

    def run_episode(self, env: VRPEnvironment, train: bool = True):
        obs = env.reset()
        trajectory = []  # (memberships, action)

        while True:
            action, memberships = self.select_action(obs)
            obs, reward, done, info = env.step(action)
            trajectory.append((memberships, action))
            if done:
                break

        total_dist = info["total_distance"]
        episode_ret = -total_dist  # MC return = final negative distance

        if train:
            for memberships, action in trajectory:
                fuzzy_q_update(
                    self.qtable,
                    memberships,
                    action,
                    episode_ret,
                    self.alpha,
                    self.n_actions,
                )
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.episode_distances.append(total_dist)
        self.episode_returns.append(episode_ret)
        return total_dist

    # -------------------------------------------------------------- Greedy eval

    def run_greedy_episode(self, env: VRPEnvironment) -> float:
        """Run one episode with pure exploitation (ε=0)."""
        saved = self.epsilon
        self.epsilon = 0.0
        dist = self.run_episode(env, train=False)
        self.epsilon = saved
        return dist

    # -------------------------------------------------------------- Inspection

    def print_policy(self):
        action_labels = [f"VISIT_{i+1}" for i in range(self.n_actions - 1)] + ["RETURN"]
        print("\n=== Learned Fuzzy Policy ===")
        print(f"{'Cap':<8} {'Dist':<8} {'Angle':<12} {'Spread':<8}  Best action")
        print("-" * 60)

        for combo in ALL_LABEL_COMBOS:
            qv = self.qtable[combo]
            best = int(np.argmax(qv))
            cap, dist, angle, spread = combo
            print(
                f"{cap:<8} {dist:<8} {angle:<12} {spread:<8}  "
                f"{action_labels[best]}  "
                f"[{' '.join(f'{v:+.3f}' for v in qv)}]"
            )
