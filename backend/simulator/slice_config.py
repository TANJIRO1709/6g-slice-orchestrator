"""
Slice definitions with SLA requirements.
Each slice type has different traffic patterns and QoS demands.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SliceSLA:
    """Service Level Agreement for a network slice."""
    name: str
    slice_id: int
    num_users: int
    min_throughput_mbps: float
    max_latency_ms: float
    target_reliability: float  # e.g., 0.99999 for URLLC
    priority: int              # higher = more critical
    color: str = "#888888"

    def check_throughput(self, actual_mbps: float) -> bool:
        return actual_mbps >= self.min_throughput_mbps

    def check_latency(self, actual_ms: float) -> bool:
        return actual_ms <= self.max_latency_ms

    def check_sla(self, throughput_mbps: float, latency_ms: float,
                  demand_mbps: float = None) -> Dict:
        # If there is no demand, throughput cannot be "too low" -- an idle
        # slice is not violating its SLA. Only check throughput against
        # what was actually asked for.
        if demand_mbps is not None and demand_mbps < 1e-6:
            tp_ok = True
        elif demand_mbps is not None:
            # Satisfied if we served the demand, or met the floor
            effective_target = min(self.min_throughput_mbps, demand_mbps)
            tp_ok = throughput_mbps >= effective_target * 0.95
        else:
            tp_ok = self.check_throughput(throughput_mbps)
        lat_ok = self.check_latency(latency_ms)
        return {
            "satisfied": tp_ok and lat_ok,
            "throughput_ok": tp_ok,
            "latency_ok": lat_ok,
            "throughput_gap": max(0, self.min_throughput_mbps - throughput_mbps),
            "latency_gap": max(0, latency_ms - self.max_latency_ms),
        }


def get_default_slices() -> Dict[str, SliceSLA]:
    """Return the three standard 6G slice types."""
    return {
        "eMBB": SliceSLA(
            name="eMBB",
            slice_id=0,
            num_users=20,
            min_throughput_mbps=10.0,
            max_latency_ms=15.0,
            target_reliability=0.999,
            priority=1,
            color="#3b82f6",
        ),
        "URLLC": SliceSLA(
            name="URLLC",
            slice_id=1,
            num_users=10,
            min_throughput_mbps=1.0,
            max_latency_ms=1.0,
            target_reliability=0.99999,
            priority=3,
            color="#ef4444",
        ),
        "mMTC": SliceSLA(
            name="mMTC",
            slice_id=2,
            num_users=50,
            min_throughput_mbps=0.05,
            max_latency_ms=100.0,
            target_reliability=0.99,
            priority=2,
            color="#22c55e",
        ),
    }
