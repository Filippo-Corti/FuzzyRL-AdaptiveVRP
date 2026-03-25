# training/visual_episode.py
from __future__ import annotations

import time
import pygame

import config
from simulation.breakdown_training import BreakdownTraining, MAX_STEPS
from simulation.snapshot import SimulationSnapshot, SimulationStats, AgentSnapshot
from collections import defaultdict


class VisualEpisode:
    """
    Runs a single breakdown recovery episode step-by-step,
    calling renderer.draw() at each step so you can watch it live.
    """

    def __init__(
        self,
        training: BreakdownTraining,
        renderer,  # your Renderer instance
        step_delay: float = 0.3,  # seconds between steps
        fps: int = 30,
    ):
        self.training = training
        self.renderer = renderer
        self.step_delay = step_delay
        self.fps = fps

    def run(self):
        """Run one full visual episode. Call after training.train() completes."""
        t = self.training
        t.reset_episode()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            # Handle pygame events so the window stays responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            truck = t.current_active_truck()
            obs = t.environment.get_observation(truck)
            prev_orphans = t.environment.graph.orphans_count
            available = t.available_actions(truck)

            action = t.agent.select_action(obs, available)
            delta_distance = t.apply_action(action, truck)
            delta_orphans = prev_orphans - t.environment.graph.orphans_count
            reward = t.per_step_reward(delta_distance)
            total_reward += reward

            done = t.environment.graph.is_fully_assigned
            is_last = step == MAX_STEPS - 1

            if done or is_last:
                terminal = t.terminal_reward(success=done)
                total_reward += terminal
                # Render final state
                self._render(step, action, total_reward, done)
                time.sleep(self.step_delay * 3)  # linger on the final frame
                break

            # Render this step
            self._render(step, action, total_reward, done)
            time.sleep(self.step_delay)
            t.advance_truck()

    def run_n(self, n: int):
        """Run n visual episodes back to back."""
        for i in range(n):
            print(f"Visual episode {i + 1}/{n}")
            self.run()
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render(self, step: int, action, total_reward: float, done: bool):
        env = self.training.environment
        agent = self.training.agent
        orphans = len(list(env.graph.unassigned_nodes()))

        snapshot = SimulationSnapshot(
            environment=env.get_snapshot(),
            agent=AgentSnapshot(
                memberships={},
                q_values=defaultdict(float),
                chosen_action=str(action),
                q_table_size=len(agent.q_table),
                epsilon=agent.epsilon,
            ),
            stats=SimulationStats(
                round=step,
                status="Recovery" if not done else "Recovered",
                orphans=orphans,
                total_nodes=len(env.graph.nodes),
                total_trucks=len(env.trucks),
                active_trucks=sum(
                    1 for t in env.trucks.values() if t.status.value != "broken"
                ),
                total_distance=env.compute_total_distance(),
                episode_reward=total_reward,
                truck_turn=self.training.current_truck_idx,
                last_action=str(action),
                best_solution_distance=0.0,
                last_distance=env.compute_total_distance(),
            ),
        )

        self.renderer.draw(snapshot)
        pygame.display.flip()
