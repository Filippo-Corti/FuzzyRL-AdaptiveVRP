from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from src.agents import TONNAgent, TransformerAgent
from src.ui import plot_metrics_comparison
from src.vrp import VRPEnvironmentBatch, VRPInstanceBatch


def build_testset_batch(device: torch.device) -> VRPInstanceBatch:
    """Create the fixed-parameter custom test-set batch from config."""
    return VRPInstanceBatch(
        batch_size=config.TESTSET_BATCH_SIZE,
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


def main() -> None:
	args = parse_args()

	if config.SEED is not None:
		torch.manual_seed(config.SEED)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if args.regenerate:
		batch = build_testset_batch(device=device)
		batch.save(config.CUSTOM_TESTSET_PATH)
		print(f"Regenerated and saved {batch.batch_size} instances to {config.CUSTOM_TESTSET_PATH}")
	else:
		batch = VRPInstanceBatch.load(config.CUSTOM_TESTSET_PATH, device=device)
		print(f"Loaded {batch.batch_size} instances from {config.CUSTOM_TESTSET_PATH}")

	alpha = float(args.alpha)
	tonn_agent = TONNAgent(w_d=1.0, w_u=-1.0)

	transformer_agent = TransformerAgent(
		node_features=config.TRANSFORMER_NODE_FEATURES,
		state_features=config.TRANSFORMER_STATE_FEATURES,
		d_model=config.TRANSFORMER_D_MODEL,
		device=device,
	)
	print("Using randomly initialized Transformer agent (no checkpoint found)")
	transformer_agent.eval()

	tonn_distance, tonn_lateness, tonn_combined = run_policy(
		batch=batch,
		select_action_callback=tonn_agent.select_actions,
		alpha=alpha,
	)
	transformer_distance, transformer_lateness, transformer_combined = run_policy(
		batch=batch,
		select_action_callback=transformer_agent.select_actions,
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
				"TONN combined cost": tonn_combined,
				"Transformer combined cost": transformer_combined,
			},
			title=f"Combined cost comparison (alpha={alpha:.2f})",
			y_label="Combined cost",
		)


if __name__ == "__main__":
	main()
