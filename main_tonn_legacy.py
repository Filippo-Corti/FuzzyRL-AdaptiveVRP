from __future__ import annotations

import argparse
from pathlib import Path

import torch

import config
from src.agents.tonn import TONNAgent
from src.ui import plot_vrp_instance
from src.vrp import VRPEnvironment, VRPEnvironmentBatch, VRPInstanceBatch


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


def run_agent_distances(batch: VRPInstanceBatch, agent: TONNAgent, alpha: float = 0.0) -> torch.Tensor:
	"""Run one TONN policy on the full batch and return per-instance total distances."""
	env = VRPEnvironmentBatch(instance_batch=batch, lateness_penalty_alpha=alpha)
	total_cost =env.solve(select_action_callback=agent.select_actions)
	return total_cost.detach().cpu()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate and preview custom VRP test-set instances.")
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
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if config.SEED is not None:
		torch.manual_seed(config.SEED)

	device = torch.device("cpu")
	if args.regenerate:
		batch = build_testset_batch(device=device)
		batch.save(config.CUSTOM_TESTSET_PATH)
		print(f"Regenerated and saved {batch.batch_size} instances to {config.CUSTOM_TESTSET_PATH}")
	else:
		batch = VRPInstanceBatch.load(config.CUSTOM_TESTSET_PATH, device=device)
		print(f"Loaded {batch.batch_size} instances from {config.CUSTOM_TESTSET_PATH}")

	distance_only_agent = TONNAgent(w_d=1.0, w_u=0.0)
	urgency_only_agent = TONNAgent(w_d=0.0, w_u=-1.0)
	better_agent = TONNAgent(w_d=1.0, w_u=-0.7)
  	
	print(f"Distance-Only:")
	distance_only = run_agent_distances(batch=batch, agent=distance_only_agent, alpha=0.2)
	print(f"Urgency-Only:")
	urgency_only = run_agent_distances(batch=batch, agent=urgency_only_agent, alpha=0.2)
	print(f"Better Agent:")
	better = run_agent_distances(batch=batch, agent=better_agent, alpha=0.1)

	results_path = Path(config.CUSTOM_TONN_RESULTS_PATH)
	results_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"distance_only": distance_only,
			"urgency_only": urgency_only,
			"better": better,
		},
		results_path,
	)
	print(f"Saved TONN distances to {results_path}")

	if not args.no_show:
		env_for_plot = VRPEnvironment(
			instance=batch.extract_instance(0),
			lateness_penalty_alpha=0.1,
		)
		env_for_plot.solve(select_action_callback=better_agent.select_actions)
		plot_vrp_instance(
			env_for_plot,
			title="Balanced TONN route on instance 0",
		)


if __name__ == "__main__":
	main()
