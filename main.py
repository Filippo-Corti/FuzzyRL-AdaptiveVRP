from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from src.agents import TONNAgent, TransformerAgent
from src.ui import plot_metrics_comparison
from src.vrp import VRPEnvironmentBatch, VRPInstanceBatch


def build_testset_batch(device: torch.device, batch_size: int) -> VRPInstanceBatch:
    """Create the fixed-parameter custom test-set batch from config."""
    return VRPInstanceBatch(
		batch_size=batch_size,
        num_nodes=config.NUM_NODES,
        device=device,
        depot_mode=config.ENV_DEPOT_MODE,
        node_xy_range=config.ENV_NODE_XY_RANGE,
        weight_range=config.ENV_WEIGHT_RANGE,
        W_value=config.ENV_W_FIXED,
        initial_visible_ratio=config.ENV_INITIAL_VISIBLE_RATIO,
        window_length_range=config.ENV_WINDOW_LENGTH_RANGE,
        cluster_count_range=config.ENV_CLUSTER_COUNT_RANGE,
        outlier_count_range=config.ENV_OUTLIER_COUNT_RANGE,
        cluster_std_range=config.ENV_CLUSTER_STD_RANGE,
    )


def run_policy(
	batch: VRPInstanceBatch,
	select_action_callback,
	alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Run one policy callback and return distance, lateness, and combined cost tensors."""
	env = VRPEnvironmentBatch(instance_batch=batch, lateness_penalty_alpha=alpha)
	combined_cost = env.solve(select_action_callback=select_action_callback).detach().cpu()
	return env.total_distance.detach().cpu(), env.total_lateness.detach().cpu(), combined_cost


def print_summary(name: str, distance: torch.Tensor, lateness: torch.Tensor, combined: torch.Tensor) -> None:
	print(f"{name}:")
	print(f"Average distance: {distance.mean().item():.2f}")
	print(f"Average lateness: {lateness.mean().item():.2f}")
	print(f"Average combined cost: {combined.mean().item():.2f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate TONN and Transformer on custom VRP dataset.")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="transformer-3200",
		help="Transformer checkpoint path or stem name (e.g. transformer-3200).",
	)
	parser.add_argument(
		"--testset-size",
		type=int,
		default=500,
		help="Number of instances in evaluation test set.",
	)
	parser.add_argument(
		"--regenerate",
		action="store_true",
		help="Regenerate the custom dataset from config and overwrite the saved file before evaluation.",
	)
	parser.add_argument(
		"--no-show",
		action="store_true",
		help="Skip matplotlib visualization.",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=0.2,
		help="Lateness penalty alpha used for combined cost.",
	)
	return parser.parse_args()


def resolve_checkpoint_path(raw_value: str) -> Path:
	"""Resolve checkpoint from full path or bare stem name."""
	p = Path(raw_value)
	if p.suffix:
		return p
	return Path("checkpoints") / f"{raw_value}.pt"


def main() -> None:
	args = parse_args()

	if config.SEED is not None:
		torch.manual_seed(config.SEED)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	requested_size = int(args.testset_size)
	if args.regenerate:
		batch = build_testset_batch(device=device, batch_size=requested_size)
		batch.save(config.CUSTOM_TESTSET_PATH)
		print(f"Regenerated and saved {batch.batch_size} instances to {config.CUSTOM_TESTSET_PATH}")
	else:
		batch = VRPInstanceBatch.load(config.CUSTOM_TESTSET_PATH, device=device)
		if batch.batch_size != requested_size:
			print(
				f"Loaded dataset has size {batch.batch_size}, expected {requested_size}. "
				"Regenerating test set with requested size."
			)
			batch = build_testset_batch(device=device, batch_size=requested_size)
			batch.save(config.CUSTOM_TESTSET_PATH)
			print(f"Saved regenerated {batch.batch_size}-instance test set to {config.CUSTOM_TESTSET_PATH}")
		else:
			print(f"Loaded {batch.batch_size} instances from {config.CUSTOM_TESTSET_PATH}")

	alpha = float(args.alpha)
	tonn_agent = TONNAgent(w_d=1.0, w_u=-1.0)

	checkpoint_path = resolve_checkpoint_path(args.checkpoint)
	if checkpoint_path.exists():
		transformer_agent = TransformerAgent.load(checkpoint_path, device=device)
		print(f"Loaded Transformer checkpoint from {checkpoint_path}")
	else:
		raise FileNotFoundError(
			f"Transformer checkpoint not found: {checkpoint_path}. "
			"Pass --checkpoint with a valid file or stem name."
		)
	transformer_agent.eval()

	tonn_distance, tonn_lateness, tonn_combined = run_policy(
		batch=batch,
		select_action_callback=tonn_agent.select_actions,
		alpha=alpha,
	)
	transformer_distance, transformer_lateness, transformer_combined = run_policy(
		batch=batch,
		select_action_callback=lambda env: transformer_agent.select_actions(env, greedy=True),
		alpha=alpha,
	)

	print_summary("TONN", tonn_distance, tonn_lateness, tonn_combined)
	print_summary("Transformer", transformer_distance, transformer_lateness, transformer_combined)

	results_path = Path(config.CUSTOM_TONN_RESULTS_PATH)
	results_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"alpha": alpha,
			"tonn_distance": tonn_distance,
			"tonn_lateness": tonn_lateness,
			"tonn_combined": tonn_combined,
			"transformer_distance": transformer_distance,
			"transformer_lateness": transformer_lateness,
			"transformer_combined": transformer_combined,
		},
		results_path,
	)
	print(f"Saved comparison results to {results_path}")

	if not args.no_show:
		plot_metrics_comparison(
			metrics={
				"TONN distance": tonn_distance,
				"Transformer distance": transformer_distance,
			},
			title="Distance comparison",
			y_label="Total distance",
		)
		plot_metrics_comparison(
			metrics={
				"TONN lateness": tonn_lateness,
				"Transformer lateness": transformer_lateness,
			},
			title="Lateness comparison",
			y_label="Total lateness",
		)
		plot_metrics_comparison(
			metrics={
				"TONN combined cost": tonn_combined,
				"Transformer combined cost": transformer_combined,
			},
			title=f"Combined cost comparison (alpha={alpha:.2f})",
			y_label="Combined cost",
		)


if __name__ == "__main__":
	main()
