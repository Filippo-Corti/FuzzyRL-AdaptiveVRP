"""
Fuzzy system for the VRP agent.

Four state variables → 3 labels each → 3^4 = 81 rule combinations.
With k_nearest=3 actions + 1 return = 4 actions: 324 Q-values total.
Still trivially small and fast to fill.

Variables
---------
capacity  : remaining_capacity_frac  ∈ [0, 1]
dist      : nearest_dist             ∈ [0, √2]
angle     : angle_to_nearest (cos)   ∈ [-1, 1]
spread    : second_vs_first_dist     ∈ [1, 5]
            1 → candidates equidistant (no clear winner)
            5 → nearest node is much closer than second (clear greedy choice)
"""

import numpy as np
from itertools import product
from typing import Dict, List, Tuple

# ------------------------------------------------------------------ MFs


def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def trapmf(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 1.0


# ------------------------------------------------------------------ Variable definitions

CAPACITY_MFS = [
    ("LOW", lambda x: trapmf(x, 0.0, 0.0, 0.25, 0.45)),
    ("MEDIUM", lambda x: trimf(x, 0.25, 0.5, 0.75)),
    ("HIGH", lambda x: trapmf(x, 0.55, 0.75, 1.0, 1.0)),
]

DIST_MFS = [
    ("NEAR", lambda x: trapmf(x, 0.00, 0.00, 0.15, 0.35)),
    ("MEDIUM", lambda x: trimf(x, 0.15, 0.35, 0.65)),
    ("FAR", lambda x: trapmf(x, 0.45, 0.65, 1.42, 1.42)),
]

ANGLE_MFS = [
    ("BACKWARDS", lambda x: trapmf(x, -1.0, -1.0, -0.25, 0.15)),
    ("SIDEWAYS", lambda x: trimf(x, -0.5, 0.0, 0.5)),
    ("ALIGNED", lambda x: trapmf(x, 0.1, 0.4, 1.0, 1.0)),
]

SPREAD_MFS = [
    ("TIGHT", lambda x: trapmf(x, 1.0, 1.0, 1.4, 2.0)),  # candidates close together
    ("MEDIUM", lambda x: trimf(x, 1.4, 2.2, 3.2)),
    ("WIDE", lambda x: trapmf(x, 2.5, 3.5, 5.0, 5.0)),  # clear nearest winner
]

CAPACITY_LABELS = [n for n, _ in CAPACITY_MFS]
DIST_LABELS = [n for n, _ in DIST_MFS]
ANGLE_LABELS = [n for n, _ in ANGLE_MFS]
SPREAD_LABELS = [n for n, _ in SPREAD_MFS]

ALL_LABEL_COMBOS = list(
    product(CAPACITY_LABELS, DIST_LABELS, ANGLE_LABELS, SPREAD_LABELS)
)


# ------------------------------------------------------------------ Fuzzification


def fuzzify(obs: dict) -> Dict[str, Dict[str, float]]:
    cap = obs["remaining_capacity_frac"]
    dist = obs["nearest_dist"]
    angle = obs["angle_to_nearest"]
    spread = obs["second_vs_first_dist"]

    return {
        "capacity": {l: fn(cap) for l, fn in CAPACITY_MFS},
        "distance": {l: fn(dist) for l, fn in DIST_MFS},
        "angle": {l: fn(angle) for l, fn in ANGLE_MFS},
        "spread": {l: fn(spread) for l, fn in SPREAD_MFS},
    }


# ------------------------------------------------------------------ Q-table helpers


def build_qtable(n_actions: int) -> Dict[Tuple, np.ndarray]:
    return {combo: np.zeros(n_actions) for combo in ALL_LABEL_COMBOS}


def fuzzy_q_values(qtable: Dict, memberships: Dict, n_actions: int) -> np.ndarray:
    """Weighted average Q-values across all firing rule combinations."""
    numerator = np.zeros(n_actions)
    denominator = 0.0

    cap_items = [(l, m) for l, m in memberships["capacity"].items() if m > 1e-9]
    dist_items = [(l, m) for l, m in memberships["distance"].items() if m > 1e-9]
    angle_items = [(l, m) for l, m in memberships["angle"].items() if m > 1e-9]
    spread_items = [(l, m) for l, m in memberships["spread"].items() if m > 1e-9]

    for cl, cm in cap_items:
        for dl, dm in dist_items:
            for al, am in angle_items:
                for sl, sm in spread_items:
                    w = cm * dm * am * sm
                    if w < 1e-9:
                        continue
                    numerator += w * qtable[(cl, dl, al, sl)]
                    denominator += w

    return numerator / denominator if denominator > 1e-9 else np.zeros(n_actions)


def fuzzy_q_update(
    qtable: Dict,
    memberships: Dict,
    action: int,
    td_target: float,
    alpha: float,
    n_actions: int,
):
    """Proportional update across all firing rules."""
    cap_items = [(l, m) for l, m in memberships["capacity"].items() if m > 1e-9]
    dist_items = [(l, m) for l, m in memberships["distance"].items() if m > 1e-9]
    angle_items = [(l, m) for l, m in memberships["angle"].items() if m > 1e-9]
    spread_items = [(l, m) for l, m in memberships["spread"].items() if m > 1e-9]

    firing = []
    total_w = 0.0

    for cl, cm in cap_items:
        for dl, dm in dist_items:
            for al, am in angle_items:
                for sl, sm in spread_items:
                    w = cm * dm * am * sm
                    if w < 1e-9:
                        continue
                    firing.append(((cl, dl, al, sl), w))
                    total_w += w

    if total_w < 1e-9:
        return

    current_q = fuzzy_q_values(qtable, memberships, n_actions)
    td_error = td_target - current_q[action]

    for key, w in firing:
        qtable[key][action] += alpha * (w / total_w) * td_error
