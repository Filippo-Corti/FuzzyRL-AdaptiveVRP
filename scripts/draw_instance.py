from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.ui.utils import plot_vrp_instance
from src.vrp import VRPEnvironment, VRPInstanceBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one VRP instance from a saved batch and draw it."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/custom/vrp_testset_200_n50.pt"),
        help="Path to a saved VRPInstanceBatch .pt file.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Instance index inside the batch.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    instance_batch = VRPInstanceBatch.load(args.dataset, device=device)

    if args.index < 0 or args.index >= instance_batch.batch_size:
        raise IndexError(
            f"index {args.index} out of range for batch_size={instance_batch.batch_size}"
        )

    instance = instance_batch.extract_instance(args.index)
    env = VRPEnvironment(instance=instance)

    plot_vrp_instance(
        env,
        title=f"VRP instance {args.index} from {args.dataset.name}",
    )


if __name__ == "__main__":
    main()
