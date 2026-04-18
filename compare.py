from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Callable, Protocol
from scipy.stats import wilcoxon

import matplotlib
import matplotlib.pyplot as plt
import torch

from src import config
from src.agents import TONNAgent, TransformerAgent
from src.agents.fuzzy import FuzzyAgent
from src.ui import plot_metrics_comparison
from src.vrp import VRPEnvironmentBatch, VRPInstanceBatch


LATENESS_ALPHA = 0.2
DATASET_FROM_FILE = False
DATASET_PATH = Path("datasets/custom/vrp_testset_200_n30.pt")
TESTSET_SIZE = 500
TESTSET_NODES = 30
RESULTS_DIR = Path("tests")
BOXPLOT_FIGSIZE = (4, 3.8)
BOXPLOT_DPI = 1200


class _PolicyAgent(Protocol):
    def select_actions(self, env: VRPEnvironmentBatch) -> torch.Tensor: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selected agents on the same VRP test set."
    )
    parser.add_argument(
        "--tonn",
        action="store_true",
        help="Include TONN baseline in comparison.",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        help="Transformer checkpoint path or stem (without extension).",
    )
    parser.add_argument(
        "--fuzzy",
        type=str,
        help="Fuzzy checkpoint path or stem (without extension).",
    )
    return parser.parse_args()


def resolve_checkpoint(path_or_stem: str, suffix: str) -> Path:
    p = Path(path_or_stem)
    if p.suffix == "":
        p = p.with_suffix(suffix)
    return p


def _evaluate_agent(
    *,
    env: VRPEnvironmentBatch,
    name: str,
    policy: Callable[[VRPEnvironmentBatch], torch.Tensor],
    lateness_penalty_alpha: float,
) -> dict[str, torch.Tensor]:
    env.reset()
    env.solve(select_action_callback=policy)

    lateness = env.total_lateness.detach().to(dtype=torch.float32, device="cpu")
    distance = env.total_distance.detach().to(dtype=torch.float32, device="cpu")
    combined = distance + lateness_penalty_alpha * lateness

    print(
        f"[{name}] mean_distance={distance.mean().item():.4f} "
        f"mean_lateness={lateness.mean().item():.4f} "
        f"mean_combined={combined.mean().item():.4f}"
    )

    return {
        "distance": distance,
        "lateness": lateness,
        "combined": combined,
    }


def _save_boxplot(
    values_by_agent: dict[str, torch.Tensor],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    labels = list(values_by_agent.keys())
    data = [values_by_agent[label].tolist() for label in labels]

    fig, ax = plt.subplots(figsize=BOXPLOT_FIGSIZE)
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, notch=True)
    palette = [
        "#0f5dff",
        "#ff4d4d",
        "#1ba45f",
        "#ff962e",
        "#8c5eff",
        "#ff2d85",
    ]

    for patch, color in zip(bp["boxes"], itertools.cycle(palette)):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=BOXPLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _run_wilcoxon_pairwise(combined_by_agent: dict[str, torch.Tensor]) -> None:
    labels = list(combined_by_agent.keys())
    if len(labels) < 2:
        return

    print("\nPairwise Wilcoxon p-values (combined metric):")
    for a, b in itertools.combinations(labels, 2):
        x = combined_by_agent[a]
        y = combined_by_agent[b]
        res = wilcoxon(x, y, zero_method="wilcox")
        print(f"- {a} vs {b}: p={res.pvalue:.6g}")


def main() -> None:
    args = parse_args()
    selected: list[tuple[str, _PolicyAgent]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.tonn:
        selected.append(("TONN", TONNAgent()))

    if args.transformer:
        ckpt_path = resolve_checkpoint(args.transformer, ".pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Transformer checkpoint not found: {ckpt_path}")
        selected.append(
            (
                "Transformer",
                TransformerAgent.load(
                    ckpt_path,
                    device=device,
                ),
            )
        )

    if args.fuzzy:
        ckpt_path = resolve_checkpoint(args.fuzzy, ".pkl")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fuzzy checkpoint not found: {ckpt_path}")
        selected.append(("Fuzzy", FuzzyAgent.load(ckpt_path, device=device)))

    if not selected:
        raise ValueError("Select at least one agent with --tonn and/or checkpoint flags.")

    if DATASET_FROM_FILE:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
        instance_batch = VRPInstanceBatch.load(DATASET_PATH, device=device)
        print(f"Loaded test dataset from {DATASET_PATH}")
    else:
        instance_batch = VRPInstanceBatch(
            batch_size=TESTSET_SIZE,
            num_nodes=TESTSET_NODES,
            depot_mode=config.ENV_DEPOT_MODE,
            node_xy_range=config.ENV_NODE_XY_RANGE,
            weight_range=config.ENV_WEIGHT_RANGE,
            W_value=config.ENV_W_FIXED,
            initial_visible_ratio=config.ENV_INITIAL_VISIBLE_RATIO,
            window_length_range=config.ENV_WINDOW_LENGTH_RANGE,
            cluster_count_range=config.ENV_CLUSTER_COUNT_RANGE,
            outlier_count_range=config.ENV_OUTLIER_COUNT_RANGE,
            cluster_std_range=config.ENV_CLUSTER_STD_RANGE,
            device=device,
        )
        print(f"Generated test dataset with {TESTSET_SIZE} instances")

    env = VRPEnvironmentBatch(
        instance_batch=instance_batch,
        lateness_penalty_alpha=LATENESS_ALPHA,
    )

    distance_by_agent: dict[str, torch.Tensor] = {}
    lateness_by_agent: dict[str, torch.Tensor] = {}
    combined_by_agent: dict[str, torch.Tensor] = {}

    for name, agent in selected:
        metrics = _evaluate_agent(
            env=env,
            name=name,
            policy=agent.select_actions,
            lateness_penalty_alpha=LATENESS_ALPHA,
        )
        distance_by_agent[name] = metrics["distance"]
        lateness_by_agent[name] = metrics["lateness"]
        combined_by_agent[name] = metrics["combined"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metrics_comparison(
        metrics=combined_by_agent,
        title=f"Combined Cost (distance + {LATENESS_ALPHA} * lateness)",
        y_label="Combined cost",
    )

    _save_boxplot(
        values_by_agent=combined_by_agent,
        ylabel="Combined cost",
        title=f"Combined cost boxplot (distance + {LATENESS_ALPHA} * lateness)",
        output_path=RESULTS_DIR / "boxplot_combined_cost.png",
    )
    _save_boxplot(
        values_by_agent=lateness_by_agent,
        ylabel="Lateness",
        title="Lateness boxplot",
        output_path=RESULTS_DIR / "boxplot_lateness.png",
    )
    _save_boxplot(
        values_by_agent=distance_by_agent,
        ylabel="Distance",
        title="Distance boxplot",
        output_path=RESULTS_DIR / "boxplot_distance.png",
    )

    _run_wilcoxon_pairwise(combined_by_agent=combined_by_agent)

    print(f"Saved boxplots in {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
