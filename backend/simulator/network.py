"""
Core 6G Network Simulator.

This is the heart of the project. It models a single base station
serving three network slices, computes per-slice metrics based on
resource allocation decisions, and tracks SLA compliance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from simulator.slice_config import SliceSLA, get_default_slices
from simulator.channel import ChannelModel
from simulator.traffic import TrafficGenerator


@dataclass
class SliceState:
    """Current state of a single network slice."""
    name: str
    allocation_ratio: float       # fraction of total BW allocated (0-1)
    bandwidth_mhz: float          # actual bandwidth in MHz
    throughput_mbps: float = 0.0  # achieved avg throughput
    latency_ms: float = 0.0      # achieved avg latency
    demand_mbps: float = 0.0     # current traffic demand
    utilization: float = 0.0     # how much of allocated BW is used
    sla_satisfied: bool = True
    violation_count: int = 0     # cumulative SLA violations
    queue_length: int = 0        # packets waiting in buffer
    avg_snr_db: float = 0.0


class NetworkSimulator:
    """
    Simulates a 6G network with multiple slices.
    
    At each step:
    1. Channel conditions update (fading changes)
    2. Traffic arrives per slice
    3. Resources are allocated (by the DRL agent or baselines)
    4. Throughput and latency are computed per slice
    5. SLA compliance is checked
    6. Metrics are recorded
    """

    def __init__(
        self,
        total_bandwidth_mhz: float = 100.0,
        seed: int = 42,
    ):
        self.total_bw = total_bandwidth_mhz
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Initialize slice definitions
        self.slices: Dict[str, SliceSLA] = get_default_slices()
        self.slice_names: List[str] = list(self.slices.keys())
        self.num_slices = len(self.slice_names)

        # Initialize channel model
        num_users = {name: s.num_users for name, s in self.slices.items()}
        self.channel = ChannelModel(num_users, seed=seed)

        # Initialize traffic generator
        self.traffic = TrafficGenerator(seed=seed + 1)

        # Current allocation (start with equal split)
        self.allocation = np.ones(self.num_slices) / self.num_slices

        # State tracking
        self.slice_states: Dict[str, SliceState] = {}
        for name in self.slice_names:
            bw = self.total_bw * self.allocation[self.slices[name].slice_id]
            self.slice_states[name] = SliceState(
                name=name,
                allocation_ratio=float(self.allocation[self.slices[name].slice_id]),
                bandwidth_mhz=bw,
            )

        # History for dashboard
        self.step_count = 0
        self.history: List[Dict] = []
        self.cumulative_violations = {name: 0 for name in self.slice_names}

    def reset(self, seed: int = None) -> Dict:
        """Reset simulator to initial state. Returns initial observation."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.channel = ChannelModel(
            {name: s.num_users for name, s in self.slices.items()},
            seed=self.seed,
        )
        self.traffic = TrafficGenerator(seed=self.seed + 1)
        self.allocation = np.ones(self.num_slices) / self.num_slices
        self.step_count = 0
        self.history = []
        self.cumulative_violations = {name: 0 for name in self.slice_names}

        # Initialize slice states
        for name in self.slice_names:
            sid = self.slices[name].slice_id
            bw = self.total_bw * self.allocation[sid]
            self.slice_states[name] = SliceState(
                name=name, allocation_ratio=float(self.allocation[sid]),
                bandwidth_mhz=bw,
            )

        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one simulation step.
        
        Args:
            action: Resource allocation ratios [eMBB, URLLC, mMTC], should sum to ~1.
                    Will be normalized internally via softmax.
        
        Returns:
            observation, reward, done, info
        """
        self.step_count += 1

        # 1. Normalize action to valid allocation (softmax)
        exp_a = np.exp(action - np.max(action))
        self.allocation = exp_a / exp_a.sum()

        # 2. Update channel conditions
        self.channel.step()

        # 3. Generate traffic
        current_traffic = self.traffic.step(self.slices)

        # 4. Compute per-slice performance
        total_reward = 0.0
        step_record = {"step": self.step_count, "slices": {}}

        for name in self.slice_names:
            sla = self.slices[name]
            sid = sla.slice_id
            alloc = float(self.allocation[sid])

            # Bandwidth allocated to this slice
            bw_mhz = self.total_bw * alloc

            # Per-user achievable rate (Shannon capacity with allocated BW)
            capacity_per_user = self.channel.get_capacity_mbps(name, bw_mhz / sla.num_users)

            # Traffic demand
            demand = current_traffic[name]["demand_mbps"]

            # Actual throughput = min(capacity, demand) per user
            actual_throughput = np.minimum(capacity_per_user, demand)
            avg_throughput = float(np.mean(actual_throughput))
            total_throughput = float(np.sum(actual_throughput))

            # Latency model: inversely related to spare capacity
            # More spare capacity = lower latency
            load_ratio = np.clip(demand / (capacity_per_user + 1e-6), 0, 2)
            base_latency = np.where(
                load_ratio < 1.0,
                0.5 + 0.5 * load_ratio,          # under-loaded: 0.5-1ms
                1.0 + 10.0 * (load_ratio - 1.0),  # overloaded: latency spikes
            )
            avg_latency = float(np.mean(base_latency))

            # Queue: packets that couldn't be served
            unserved = np.maximum(0, demand - capacity_per_user)
            queue = int(np.sum(unserved > 0))

            # Utilization
            total_capacity = float(np.sum(capacity_per_user))
            utilization = float(np.sum(actual_throughput)) / (total_capacity + 1e-6)

            # SLA check
            sla_result = sla.check_sla(avg_throughput, avg_latency,
                                       demand_mbps=float(np.mean(demand)))
            if not sla_result["satisfied"]:
                self.cumulative_violations[name] += 1

            # Update slice state
            self.slice_states[name] = SliceState(
                name=name,
                allocation_ratio=alloc,
                bandwidth_mhz=bw_mhz,
                throughput_mbps=avg_throughput,
                latency_ms=avg_latency,
                demand_mbps=float(np.mean(demand)),
                utilization=min(utilization, 1.0),
                sla_satisfied=sla_result["satisfied"],
                violation_count=self.cumulative_violations[name],
                queue_length=queue,
                avg_snr_db=self.channel.get_avg_snr_db(name),
            )

            # Reward contribution from this slice
            if not sla_result["satisfied"]:
                if name == "URLLC":
                    total_reward -= 10.0
                elif name == "eMBB":
                    total_reward -= 2.0
                else:
                    total_reward -= 1.0
            else:
                total_reward += 0.5  # bonus for satisfied SLA

            # Utilization bonus
            total_reward += utilization * 0.3

            step_record["slices"][name] = {
                "allocation": round(alloc, 4),
                "throughput_mbps": round(avg_throughput, 2),
                "latency_ms": round(avg_latency, 3),
                "demand_mbps": round(float(np.mean(demand)), 2),
                "utilization": round(min(utilization, 1.0), 3),
                "sla_satisfied": sla_result["satisfied"],
                "violations_total": self.cumulative_violations[name],
                "queue_length": queue,
                "avg_snr_db": round(self.channel.get_avg_snr_db(name), 1),
            }

        # Stability bonus: penalize drastic allocation changes
        if len(self.history) > 0:
            prev_alloc = np.array([
                self.history[-1]["slices"][n]["allocation"]
                for n in self.slice_names
            ])
            change = np.sum(np.abs(self.allocation - prev_alloc))
            total_reward -= change * 0.5  # small penalty for instability

        step_record["reward"] = round(total_reward, 3)
        step_record["allocation"] = {
            name: round(float(self.allocation[self.slices[name].slice_id]), 4)
            for name in self.slice_names
        }
        self.history.append(step_record)

        # Keep history bounded
        if len(self.history) > 2000:
            self.history = self.history[-1000:]

        obs = self._get_observation()
        done = False  # continuous simulation, never terminates
        info = step_record

        return obs, total_reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector for the DRL agent.
        
        Observation space (per slice × 3 slices):
        - Current allocation ratio
        - Average throughput (normalized)
        - Average latency (normalized)
        - Demand (normalized)
        - Utilization
        - SLA satisfied (0 or 1)
        - Queue length (normalized)
        Plus channel state vector (3 values per slice)
        
        Total: 7 * 3 + 3 * 3 = 30 dimensions
        """
        obs = []
        for name in self.slice_names:
            s = self.slice_states[name]
            sla = self.slices[name]
            obs.extend([
                s.allocation_ratio,
                s.throughput_mbps / max(sla.min_throughput_mbps, 1.0),  # normalized
                s.latency_ms / max(sla.max_latency_ms, 1.0),          # normalized
                s.demand_mbps / max(sla.min_throughput_mbps, 1.0),     # normalized
                s.utilization,
                1.0 if s.sla_satisfied else 0.0,
                min(s.queue_length / 50.0, 1.0),  # normalize queue
            ])

        # Append channel state
        obs.extend(self.channel.get_state_vector().tolist())

        return np.array(obs, dtype=np.float32)

    def get_metrics_snapshot(self) -> Dict:
        """Get current metrics for the dashboard."""
        return {
            "step": self.step_count,
            "slices": {
                name: {
                    "allocation": self.slice_states[name].allocation_ratio,
                    "throughput_mbps": self.slice_states[name].throughput_mbps,
                    "latency_ms": self.slice_states[name].latency_ms,
                    "demand_mbps": self.slice_states[name].demand_mbps,
                    "utilization": self.slice_states[name].utilization,
                    "sla_satisfied": self.slice_states[name].sla_satisfied,
                    "violations_total": self.slice_states[name].violation_count,
                    "queue_length": self.slice_states[name].queue_length,
                    "avg_snr_db": self.slice_states[name].avg_snr_db,
                    "color": self.slices[name].color,
                }
                for name in self.slice_names
            },
            "total_violations": sum(self.cumulative_violations.values()),
        }
