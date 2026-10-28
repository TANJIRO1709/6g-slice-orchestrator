"""
Gymnasium environment wrapper for the 6G Network Simulator.
This is what Stable-Baselines3 interacts with during training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simulator.network import NetworkSimulator


class NetworkSliceEnv(gym.Env):
    """
    Gym environment for 6G network slice resource allocation.
    
    Observation: 30-dim vector (see NetworkSimulator._get_observation)
    Action: 3-dim continuous vector (allocation ratios, normalized via softmax)
    Reward: Weighted SLA compliance + utilization
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_bandwidth_mhz: float = 100.0,
        max_steps: int = 1000,
        seed: int = 42,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.sim = NetworkSimulator(total_bandwidth_mhz=total_bandwidth_mhz, seed=seed)

        # Observation space: 30 dimensions (7 per slice * 3 + 9 channel features)
        obs_dim = 7 * 3 + 3 * 3  # = 30
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space: 3 continuous values (will be softmax-normalized)
        # Using range [-1, 1] for stable training, mapped to allocation via softmax
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.current_step = 0

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        if seed is not None:
            obs = self.sim.reset(seed=seed)
        else:
            obs = self.sim.reset()
        self.current_step = 0
        return obs, {}

    def step(self, action):
        """Execute one step."""
        self.current_step += 1
        obs, reward, _, info = self.sim.step(action)

        terminated = False
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Print current state."""
        metrics = self.sim.get_metrics_snapshot()
        print(f"\n=== Step {metrics['step']} ===")
        for name, data in metrics["slices"].items():
            status = "✓" if data["sla_satisfied"] else "✗"
            print(
                f"  {name:6s} | alloc: {data['allocation']:.1%} | "
                f"tp: {data['throughput_mbps']:7.1f} Mbps | "
                f"lat: {data['latency_ms']:5.2f} ms | "
                f"util: {data['utilization']:.1%} | "
                f"SLA: {status} | violations: {data['violations_total']}"
            )


# Register the environment so it can be created with gym.make()
gym.register(
    id="NetworkSlice-v0",
    entry_point="simulator.env:NetworkSliceEnv",
    max_episode_steps=1000,
)
