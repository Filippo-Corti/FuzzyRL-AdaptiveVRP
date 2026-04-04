import torch
import time
from pathlib import Path

from agent import transformer_agent
from env.batch_env import BatchVRPEnv


def train(
    num_episodes: int = 10_000,
    batch_size: int = 64,
    num_nodes: int = 15,
    d_model: int = 128,
    save_every: int = 500,
    save_path: str = "checkpoints/transformer.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = transformer.TransformerAgent(
        node_features=4, state_features=3, d_model=d_model, device=device
    )
    env = BatchVRPEnv(batch_size=batch_size, num_nodes=num_nodes, device=device)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Training: {num_episodes} episodes, batch={batch_size}, nodes={num_nodes}")

    for episode in range(1, num_episodes + 1):
        t0 = time.time()

        # --- Greedy baseline rollout (no gradients) ---
        env.reset()
        baseline_rewards = torch.zeros(batch_size, device=device)

        with torch.no_grad():
            while not env.all_done():
                node_features, truck_state, mask = env.get_state()
                actions, _ = agent.select_action(
                    node_features, truck_state, mask, greedy=True
                )
                rewards = env.step(actions)
                baseline_rewards += rewards

        # --- Sampled rollout (with gradients) ---
        env.reset()
        sampled_rewards = torch.zeros(batch_size, device=device)
        log_probs_list: list[torch.Tensor] = []  # each: (B,)

        while not env.all_done():
            node_features, truck_state, mask = env.get_state()
            actions, log_probs = agent.select_action(
                node_features, truck_state, mask, greedy=False
            )
            rewards = env.step(actions)
            sampled_rewards += rewards
            log_probs_list.append(log_probs)

        # --- REINFORCE update ---
        # advantage: (B,) — positive means sampled beat greedy
        advantage = sampled_rewards - baseline_rewards  # (B,)

        # sum log probs across steps: (B,)
        log_probs_total = torch.stack(log_probs_list, dim=0).sum(dim=0)  # (B,)

        # loss: negative because we maximise expected return
        loss = -(advantage.detach() * log_probs_total).mean()

        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(agent.encoder.parameters()) + list(agent.decoder.parameters()),
            max_norm=1.0,
        )
        agent.optimizer.step()

        elapsed = time.time() - t0

        if episode % 50 == 0:
            print(
                f"Episode {episode:5d} | "
                f"sampled={sampled_rewards.mean():.3f} | "
                f"baseline={baseline_rewards.mean():.3f} | "
                f"adv={advantage.mean():.3f} | "
                f"loss={loss.item():.4f} | "
                f"{elapsed:.2f}s"
            )

        if episode % save_every == 0:
            torch.save(
                {
                    "episode": episode,
                    "encoder": agent.encoder.state_dict(),
                    "decoder": agent.decoder.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                },
                save_path,
            )
            print(f"Saved checkpoint to {save_path}")

    return agent


if __name__ == "__main__":
    train()
