"""
DRL training script for the network slice allocator.

Usage:
    python -m rl.train --timesteps 500000
    python -m rl.train --timesteps 100000 --eval    # quick test run
"""

import argparse
import os
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from simulator.env import NetworkSliceEnv


class MetricsCallback(BaseCallback):
    """Logs custom metrics during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_violations = []

    def _on_step(self) -> bool:
        # Log info from the environment
        for info in self.locals.get("infos", []):
            if "slices" in info:
                violations = sum(
                    1 for s in info["slices"].values()
                    if not s.get("sla_satisfied", True)
                )
                self.episode_violations.append(violations)

        # Every 10k steps, print summary
        if self.num_timesteps % 10000 == 0 and self.episode_violations:
            avg_viol = np.mean(self.episode_violations[-1000:])
            print(f"  Step {self.num_timesteps}: avg violations/step = {avg_viol:.2f}")
            self.episode_violations = []

        return True


def create_env(seed: int = 42, max_steps: int = 1000) -> NetworkSliceEnv:
    """Create and wrap the environment."""
    env = NetworkSliceEnv(
        total_bandwidth_mhz=100.0,
        max_steps=max_steps,
        seed=seed,
    )
    return Monitor(env)


def train(timesteps: int = 500_000, eval_mode: bool = False):
    """Train the PPO agent."""
    print("=" * 60)
    print("  6G Network Slice Orchestrator — DRL Training")
    print("=" * 60)

    # Create directories
    models_dir = Path("models")
    logs_dir = Path("logs")
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Training environment
    train_env = create_env(seed=42)

    # Evaluation environment (different seed for generalization)
    eval_env = create_env(seed=123)

    # PPO with custom network architecture
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(logs_dir),
        policy_kwargs={
            "net_arch": {
                "pi": [128, 128, 64],   # actor network
                "vf": [128, 128, 64],   # critic network
            },
        },
    )

    print(f"\nTraining for {timesteps:,} timesteps...")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}\n")

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(logs_dir / "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    metrics_callback = MetricsCallback()

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, metrics_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = models_dir / "ppo_slice_allocator_final"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}")

    # Quick evaluation
    if eval_mode:
        evaluate(model, eval_env, episodes=10)

    train_env.close()
    eval_env.close()


def evaluate(model, env, episodes: int = 10):
    """Evaluate the trained model."""
    print("\n" + "=" * 60)
    print("  Evaluation Results")
    print("=" * 60)

    all_rewards = []
    all_violations = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        ep_violations = 0
        steps = 0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if "slices" in info:
                for s in info["slices"].values():
                    if not s.get("sla_satisfied", True):
                        ep_violations += 1

        all_rewards.append(total_reward)
        all_violations.append(ep_violations)
        print(f"  Episode {ep + 1}: reward = {total_reward:.1f}, violations = {ep_violations}")

    print(f"\n  Avg Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"  Avg Violations: {np.mean(all_violations):.1f} ± {np.std(all_violations):.1f}")


def run_baselines(episodes: int = 10):
    """Run baseline strategies for comparison."""
    print("\n" + "=" * 60)
    print("  Baseline Comparisons")
    print("=" * 60)

    env = create_env(seed=123, max_steps=500)

    baselines = {
        "Equal Static": lambda obs: np.array([0.0, 0.0, 0.0]),  # softmax → equal
        "eMBB Heavy": lambda obs: np.array([1.0, 0.0, -1.0]),   # favor eMBB
        "URLLC Heavy": lambda obs: np.array([0.0, 1.0, -1.0]),  # favor URLLC
    }

    for name, policy_fn in baselines.items():
        rewards = []
        violations = []

        for ep in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            ep_violations = 0
            done = False

            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                if "slices" in info:
                    for s in info["slices"].values():
                        if not s.get("sla_satisfied", True):
                            ep_violations += 1

            rewards.append(total_reward)
            violations.append(ep_violations)

        print(f"\n  {name}:")
        print(f"    Avg Reward:     {np.mean(rewards):8.1f} ± {np.std(rewards):.1f}")
        print(f"    Avg Violations: {np.mean(violations):8.1f} ± {np.std(violations):.1f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL agent for 6G slice allocation")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparisons only")
    args = parser.parse_args()

    if args.baselines:
        run_baselines()
    else:
        train(timesteps=args.timesteps, eval_mode=args.eval)
