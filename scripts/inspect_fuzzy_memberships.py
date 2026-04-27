from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.fuzzy import FuzzyAgent, FuzzyFeature


METRIC_TO_ATTR = {
    "distance": "mf_distance",
    "urgency": "mf_urgency",
    "demand_ratio": "mf_demand_ratio",
    "cluster_density": "mf_cluster_density",
    "detour_cost": "mf_detour_cost",
}

FUZZY_CHECKPOINT_RE = re.compile(r"^fuzzy-(?P<step>\d+)\.pkl$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect fuzzy membership functions for a fresh agent and the latest "
            "checkpointed agent, and print top rules."
        )
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_TO_ATTR.keys()),
        default="distance",
        help="Which fuzzy feature to plot.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing fuzzy-<step>.pkl checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("img"),
        help="Directory where PDF plots will be saved.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Torch device used to create/load agents.",
    )
    return parser.parse_args()


def _find_latest_fuzzy_checkpoint(checkpoints_dir: Path) -> tuple[Path, int]:
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    latest: tuple[Path, int] | None = None
    for path in checkpoints_dir.iterdir():
        if not path.is_file():
            continue
        match = FUZZY_CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        if latest is None or step > latest[1]:
            latest = (path, step)

    if latest is None:
        raise FileNotFoundError(
            f"No fuzzy checkpoints matching fuzzy-<step>.pkl found in {checkpoints_dir}"
        )

    return latest


def _plot_feature_memberships(
    *,
    feature: FuzzyFeature,
    metric_name: str,
    title: str,
    output_path: Path,
) -> None:
    x = torch.linspace(0.0, 1.0, 400)
    y = feature(x)

    low = y[:, 0].detach().cpu().numpy()
    medium = y[:, 1].detach().cpu().numpy()
    high = y[:, 2].detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x_np, low, label="Low", linewidth=2.0)
    ax.plot(x_np, medium, label="Medium", linewidth=2.0)
    ax.plot(x_np, high, label="High", linewidth=2.0)

    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f"{metric_name.capitalize()} value", fontsize=16)
    ax.set_ylabel("Membership degree", fontsize=16)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=16)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)


def _feature_from_agent(agent: FuzzyAgent, metric_name: str) -> FuzzyFeature:
    attr_name = METRIC_TO_ATTR[metric_name]
    feature = getattr(agent, attr_name)
    if not isinstance(feature, FuzzyFeature):
        raise TypeError(f"Attribute '{attr_name}' is not a FuzzyFeature")
    return feature


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)

    fresh_agent = FuzzyAgent(device=device)
    fresh_feature = _feature_from_agent(fresh_agent, args.metric)
    fresh_plot_path = args.output_dir / f"fuzzy_membership_{args.metric}_fresh.pdf"
    _plot_feature_memberships(
        feature=fresh_feature,
        metric_name=args.metric,
        title=f"Fresh FuzzyAgent",
        output_path=fresh_plot_path,
    )
    print(f"Saved fresh-agent membership plot: {fresh_plot_path}")

    latest_ckpt_path, latest_step = _find_latest_fuzzy_checkpoint(args.checkpoints_dir)
    latest_agent = FuzzyAgent.load(latest_ckpt_path, device=device)

    print(f"Loaded latest fuzzy checkpoint: {latest_ckpt_path} (step={latest_step})")
    print("Top 10 fuzzy rules (feature, label, weight):")
    for idx, (feature_name, label_name, weight) in enumerate(
        latest_agent.top_rules(top_k=60),
        start=1,
    ):
        print(f"{idx:2d}. ({feature_name}, {label_name}) -> {weight:+.6f}")

    latest_feature = _feature_from_agent(latest_agent, args.metric)
    latest_plot_path = args.output_dir / f"fuzzy_membership_{args.metric}_latest.pdf"
    _plot_feature_memberships(
        feature=latest_feature,
        metric_name=args.metric,
        title=f"Best FuzzyAgent",
        output_path=latest_plot_path,
    )
    print(f"Saved latest-agent membership plot: {latest_plot_path}")


if __name__ == "__main__":
    main()
