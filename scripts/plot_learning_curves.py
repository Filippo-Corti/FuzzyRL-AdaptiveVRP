from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.utils import plot_learning_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Transformer vs Fuzzy learning curves for selected metrics "
            "and save them as PDF files."
        )
    )
    parser.add_argument(
        "--transformer-csv",
        type=Path,
        default=Path("checkpoints/transformer-metrics.csv"),
        help="Path to transformer metrics CSV.",
    )
    parser.add_argument(
        "--fuzzy-csv",
        type=Path,
        default=Path("checkpoints/fuzzy-metrics.csv"),
        help="Path to fuzzy metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests"),
        help="Directory where PDF plots will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving PDFs.",
    )
    return parser.parse_args()


def _read_csv_columns(path: Path) -> dict[str, list[float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        data: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                raw = (row.get(name) or "").strip()
                if raw == "":
                    data[name].append(float("nan"))
                else:
                    data[name].append(float(raw))

    return data


def _interpolate_nan_linear(
    episodes: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if episodes.dim() != 1 or values.dim() != 1:
        raise ValueError("episodes and values must be 1D tensors")
    if episodes.shape != values.shape:
        raise ValueError("episodes and values must have the same shape")

    missing_mask = torch.isnan(values)
    if not missing_mask.any():
        return values

    known_mask = ~missing_mask
    if not known_mask.any():
        raise ValueError("Cannot interpolate: all values are NaN")

    known_idx = torch.where(known_mask)[0]
    result = values.clone()

    if known_idx.numel() == 1:
        result[:] = values[known_idx[0]]
        return result

    first_known = int(known_idx[0].item())
    last_known = int(known_idx[-1].item())

    # Keep curve constant before first and after last known checkpoint.
    result[:first_known] = values[first_known]
    result[last_known + 1 :] = values[last_known]

    for i in range(known_idx.numel() - 1):
        left = int(known_idx[i].item())
        right = int(known_idx[i + 1].item())

        x0 = episodes[left]
        x1 = episodes[right]
        y0 = values[left]
        y1 = values[right]

        if right == left + 1:
            continue

        xs = episodes[left : right + 1]
        t = (xs - x0) / (x1 - x0 + 1e-12)
        result[left : right + 1] = y0 + t * (y1 - y0)

    return result


def _metric_series(
    data: dict[str, list[float]],
    episodes: torch.Tensor,
    metric_name: str,
) -> torch.Tensor:
    if metric_name == "sampled_minus_tonn":
        sampled_minus_tonn = data.get("sampled_minus_tonn")
        if sampled_minus_tonn is not None and not torch.isnan(
            torch.tensor(sampled_minus_tonn, dtype=torch.float32)
        ).all():
            raw = torch.tensor(sampled_minus_tonn, dtype=torch.float32)
            return _interpolate_nan_linear(episodes, raw)

        sampled = data.get("sampled_cost_mean")
        tonn = data.get("tonn_cost_mean")
        if sampled is None or tonn is None:
            raise ValueError(
                "Cannot build sampled_minus_tonn: missing sampled_cost_mean or tonn_cost_mean"
            )

        sampled_t = torch.tensor(sampled, dtype=torch.float32)
        tonn_t = torch.tensor(tonn, dtype=torch.float32)
        raw = sampled_t - tonn_t
        return _interpolate_nan_linear(episodes, raw)

    values = data.get(metric_name)
    if values is None:
        raise ValueError(f"Metric '{metric_name}' not found in CSV")
    return torch.tensor(values, dtype=torch.float32)


def _plot_one_metric(
    *,
    metric_key: str,
    metric_label: str,
    transformer_data: dict[str, list[float]],
    fuzzy_data: dict[str, list[float]],
    output_dir: Path,
    show: bool,
) -> None:
    transformer_episodes = torch.tensor(transformer_data["episode"], dtype=torch.float32)
    fuzzy_episodes = torch.tensor(fuzzy_data["episode"], dtype=torch.float32)

    transformer_values = _metric_series(transformer_data, transformer_episodes, metric_key)
    fuzzy_values = _metric_series(fuzzy_data, fuzzy_episodes, metric_key)

    out_file = output_dir / f"learning_curve_{metric_key}.pdf"

    plot_learning_curves(
        curves={
            "Transformer": (transformer_episodes, transformer_values),
            "Fuzzy": (fuzzy_episodes, fuzzy_values),
        },
        title=f"{metric_label}",
        x_label="Episode",
        y_label=metric_label,
        output_path=out_file,
        show=show,
    )

    print(f"Saved {out_file}")


def main() -> None:
    args = parse_args()

    for path in (args.transformer_csv, args.fuzzy_csv):
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")

    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"

    transformer_data = _read_csv_columns(args.transformer_csv)
    fuzzy_data = _read_csv_columns(args.fuzzy_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("advantage_mean", "Mean Advantage"),
        ("sampled_cost_mean", "Mean Sampled Cost"),
        ("entropy_mean", "Mean Entropy"),
        ("sampled_minus_tonn", "Sampled - TONN"),
    ]

    for metric_key, metric_label in metrics:
        _plot_one_metric(
            metric_key=metric_key,
            metric_label=metric_label,
            transformer_data=transformer_data,
            fuzzy_data=fuzzy_data,
            output_dir=args.output_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
