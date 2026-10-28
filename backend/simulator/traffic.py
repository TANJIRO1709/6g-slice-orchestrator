"""
Traffic generators for different slice types.
Each generates realistic demand patterns that the allocator must satisfy.
"""

import numpy as np
from typing import Dict


class TrafficGenerator:
    """
    Generates per-user traffic demand for each slice.
    
    eMBB:  Bursty high-volume (video streaming, downloads)
    URLLC: Small periodic packets with strict deadlines (control, robotics)
    mMTC:  Infrequent small packets from many devices (sensors, IoT)
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.step_count = 0

        # Traffic state (for temporal patterns)
        self._embb_burst_active = False
        self._burst_remaining = 0

    def generate_embb(self, num_users: int) -> Dict:
        """
        eMBB traffic: mixture of steady streaming + random bursts.
        Returns demanded throughput per user (Mbps) and packet sizes.
        """
        # Base demand: each user wants 8-22 Mbps.
        # Calibrated so that the SLA is reachable at high allocation but
        # violated at low allocation -- this is what makes the allocation
        # decision meaningful rather than trivially impossible.
        base_demand = self.rng.uniform(6, 16, size=num_users)

        # Random burst events (simulates someone starting a 4K stream)
        burst_mask = self.rng.random(num_users) < 0.1  # 10% chance per user
        base_demand[burst_mask] *= self.rng.uniform(1.3, 2.0, size=burst_mask.sum())

        # Time-of-day pattern: sinusoidal variation
        time_factor = 0.7 + 0.3 * np.sin(2 * np.pi * self.step_count / 1000)
        demand = base_demand * time_factor

        return {
            "demand_mbps": demand,
            "total_demand_mbps": float(demand.sum()),
            "avg_demand_mbps": float(demand.mean()),
            "packet_size_bytes": self.rng.randint(500, 1500, size=num_users),
            "num_active_users": int((demand > 10).sum()),
        }

    def generate_urllc(self, num_users: int) -> Dict:
        """
        URLLC traffic: small packets, periodic, extremely latency-sensitive.
        """
        # Each user generates small control packets
        demand = self.rng.uniform(0.5, 2.0, size=num_users)

        # Some users have urgent bursts (e.g., emergency stop command)
        urgent_mask = self.rng.random(num_users) < 0.05
        demand[urgent_mask] *= 3.0

        # Deadline pressure: how close to missing latency target
        deadline_pressure = self.rng.uniform(0.3, 1.0, size=num_users)

        return {
            "demand_mbps": demand,
            "total_demand_mbps": float(demand.sum()),
            "avg_demand_mbps": float(demand.mean()),
            "packet_size_bytes": self.rng.randint(32, 256, size=num_users),
            "deadline_pressure": deadline_pressure,
            "num_urgent": int(urgent_mask.sum()),
        }

    def generate_mmtc(self, num_users: int) -> Dict:
        """
        mMTC traffic: many devices, sporadic small transmissions.
        """
        # Only a fraction of devices transmit in each step
        active_prob = 0.05 + 0.03 * np.sin(2 * np.pi * self.step_count / 500)
        active_mask = self.rng.random(num_users) < active_prob
        num_active = int(active_mask.sum())

        demand = np.zeros(num_users)
        if num_active > 0:
            demand[active_mask] = self.rng.uniform(0.01, 0.5, size=num_active)

        return {
            "demand_mbps": demand,
            "total_demand_mbps": float(demand.sum()),
            "avg_demand_mbps": float(demand.mean()) if num_active > 0 else 0.0,
            "packet_size_bytes": np.where(active_mask, self.rng.randint(10, 100, size=num_users), 0),
            "num_active_devices": num_active,
            "total_devices": num_users,
        }

    def step(self, slice_configs: Dict) -> Dict[str, Dict]:
        """Generate traffic for all slices in one time step."""
        self.step_count += 1

        traffic = {}
        for name, cfg in slice_configs.items():
            n = cfg.num_users
            if name == "eMBB":
                traffic[name] = self.generate_embb(n)
            elif name == "URLLC":
                traffic[name] = self.generate_urllc(n)
            elif name == "mMTC":
                traffic[name] = self.generate_mmtc(n)

        return traffic

    def inject_spike(self, slice_name: str, multiplier: float = 3.0):
        """Manually inject a traffic spike for testing."""
        self._embb_burst_active = True
        self._burst_remaining = 50  # lasts 50 steps
