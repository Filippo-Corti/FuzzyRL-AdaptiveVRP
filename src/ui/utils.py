from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.vrp import VRPEnvironment


def plot_vrp_instance(env: VRPEnvironment, title: str = "VRP Environment") -> None:
    """Visualize one solved/unsolved single VRP environment with its route."""
    if env.batch_size != 1:
        raise ValueError("plot_vrp_instance expects a single VRPEnvironment")

    instance = env.instance
    depot = instance.depot_xy[0].detach().cpu().numpy()
    nodes = instance.node_xy[0].detach().cpu().numpy()
    route = env.routes[0]

    plt.figure(figsize=(8, 8))
    plt.scatter(nodes[:, 0], nodes[:, 1], s=28, c="tab:blue", alpha=0.85, label="Customers")
    plt.scatter([depot[0]], [depot[1]], s=110, c="tab:red", marker="*", label="Depot")

    if len(route) >= 2:
        route_tensor = torch.tensor(route, dtype=torch.float32)
        route_xy = route_tensor.cpu().numpy()
        plt.plot(route_xy[:, 0], route_xy[:, 1], lw=1.4, c="tab:green", alpha=0.9, label="Route")

    plt.xlim(instance.node_xy_range[0], instance.node_xy_range[1])
    plt.ylim(instance.node_xy_range[0], instance.node_xy_range[1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.25)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    metrics: dict[str, torch.Tensor],
    title: str = "Metrics comparison over instances",
    x_label: str = "Instance index",
    y_label: str = "Metric value",
) -> None:
    """Plot any number of 1D metric tensors over instance index."""
    if not metrics:
        raise ValueError("metrics must contain at least one named tensor")

    first = next(iter(metrics.values()))
    if first.dim() != 1:
        raise ValueError("all metric tensors must be 1D")

    expected_shape = first.shape
    for name, values in metrics.items():
        if values.dim() != 1:
            raise ValueError(f"metric '{name}' must be a 1D tensor")
        if values.shape != expected_shape:
            raise ValueError("all metric tensors must have the same shape")

    x = torch.arange(first.numel()).cpu().numpy()

    plt.figure(figsize=(11, 5.5))
    for name, values in metrics.items():
        y = values.detach().cpu().numpy()
        plt.plot(x, y, lw=1.6, label=name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_learning_curves(
    curves: dict[str, tuple[torch.Tensor, torch.Tensor]],
    title: str = "Learning curve",
    x_label: str = "Episode",
    y_label: str = "Metric value",
    output_path: str | Path | None = None,
    show: bool = True,
    title_fontsize: int = 30,
    label_fontsize: int = 24,
    tick_fontsize: int = 20,
    legend_fontsize: int = 20,
) -> None:
    """Plot one or more learning curves and optionally save to file."""
    if not curves:
        raise ValueError("curves must contain at least one named series")

    plt.figure(figsize=(10, 5.5))

    for name, (episodes, values) in curves.items():
        if episodes.dim() != 1:
            raise ValueError(f"episodes for '{name}' must be a 1D tensor")
        if values.dim() != 1:
            raise ValueError(f"values for '{name}' must be a 1D tensor")
        if episodes.shape != values.shape:
            raise ValueError(
                f"episodes and values for '{name}' must have the same shape"
            )

        x = episodes.detach().cpu().numpy()
        y = values.detach().cpu().numpy()
        plt.plot(x, y, lw=1.5, label=name)

    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(x_label, fontsize=label_fontsize)
    plt.ylabel(y_label, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(alpha=0.25)
    plt.legend(loc="best", fontsize=legend_fontsize)
    plt.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", format="pdf", dpi=1200)

    if show:
        plt.show()

    plt.close()