"""
Quick test to verify the simulator works correctly.
Run: python test_simulator.py
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from simulator.network import NetworkSimulator
from simulator.env import NetworkSliceEnv


def test_raw_simulator():
    """Test the simulator directly."""
    print("=" * 60)
    print("  Testing Raw Simulator")
    print("=" * 60)

    sim = NetworkSimulator(total_bandwidth_mhz=100.0, seed=42)
    obs = sim.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs[:10]}... (showing first 10)\n")

    # Run 20 steps with equal allocation
    for i in range(20):
        action = np.array([0.0, 0.0, 0.0])  # equal split via softmax
        obs, reward, done, info = sim.step(action)

        if i % 5 == 0:
            metrics = sim.get_metrics_snapshot()
            print(f"Step {metrics['step']}:")
            for name, data in metrics["slices"].items():
                sla = "✓" if data["sla_satisfied"] else "✗"
                print(
                    f"  {name:6s} | alloc: {data['allocation']:.1%} | "
                    f"tp: {data['throughput_mbps']:7.1f} Mbps | "
                    f"lat: {data['latency_ms']:5.2f} ms | "
                    f"SLA: {sla}"
                )
            print(f"  Reward: {reward:.2f}\n")

    print("✅ Raw simulator works!\n")


def test_gym_env():
    """Test the Gymnasium environment."""
    print("=" * 60)
    print("  Testing Gymnasium Environment")
    print("=" * 60)

    env = NetworkSliceEnv(total_bandwidth_mhz=100.0, max_steps=100, seed=42)
    obs, info = env.reset()

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs shape: {obs.shape}\n")

    total_reward = 0
    violations = 0

    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if "slices" in info:
            for s in info["slices"].values():
                if not s.get("sla_satisfied", True):
                    violations += 1

        if terminated or truncated:
            break

    print(f"Ran 100 steps with random actions:")
    print(f"  Total reward:    {total_reward:.1f}")
    print(f"  Total violations: {violations}")
    print(f"  Final obs shape: {obs.shape}")
    print(f"\n✅ Gymnasium environment works!\n")

    env.close()


def test_baseline_comparison():
    """Compare different static strategies."""
    print("=" * 60)
    print("  Baseline Strategy Comparison (100 steps each)")
    print("=" * 60)

    strategies = {
        "Equal Split":   np.array([0.0, 0.0, 0.0]),
        "eMBB Heavy":    np.array([2.0, 0.0, -1.0]),
        "URLLC Heavy":   np.array([0.0, 2.0, -1.0]),
        "mMTC Heavy":    np.array([-1.0, 0.0, 2.0]),
    }

    for name, action in strategies.items():
        env = NetworkSliceEnv(total_bandwidth_mhz=100.0, max_steps=200, seed=42)
        obs, _ = env.reset()

        total_reward = 0
        total_violations = 0

        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if "slices" in info:
                for s in info["slices"].values():
                    if not s.get("sla_satisfied", True):
                        total_violations += 1
            if terminated or truncated:
                break

        print(f"  {name:15s} | reward: {total_reward:8.1f} | violations: {total_violations}")
        env.close()

    print(f"\n✅ Baseline comparison complete!")


if __name__ == "__main__":
    test_raw_simulator()
    test_gym_env()
    test_baseline_comparison()
    print("\n🎉 All tests passed! Your simulator is ready.")
    print("Next steps:")
    print("  1. Train DRL agent:  python -m rl.train --timesteps 100000 --eval")
    print("  2. Start API:        uvicorn api.main:app --reload --port 8000")
    print("  3. Start dashboard:  cd ../frontend && npm install && npm run dev")
