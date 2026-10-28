"""
Monitor Agent — Phase 5

The first stage of the agentic pipeline. Watches per-slice KPIs at every
simulation step and emits structured alerts when something looks wrong.

Detection is deliberately rule-based here: anomaly detection on numeric
thresholds is exactly the kind of task where an LLM adds cost without
adding accuracy. The LLM reasoning happens downstream in the Diagnosis
Agent, where context actually matters.

What this agent detects:
    - Direct SLA violations (throughput below target / latency above target)
    - Degradation trends (metric declining steadily over a window)
    - Queue buildup (congestion forming before it becomes a violation)
    - Allocation oscillation (the ping-pong anti-pattern)
    - Underutilization (over-provisioned slice wasting bandwidth)
"""

from typing import Dict, List, Optional
from collections import deque
import numpy as np

from agents.schemas import Alert, Severity, IssueType


# SLA targets per slice (mirrors simulator/slice_config.py)
SLA_TARGETS = {
    "eMBB":  {"min_throughput": 10.0,  "max_latency": 15.0},
    "URLLC": {"min_throughput": 1.0,   "max_latency": 1.0},
    "mMTC":  {"min_throughput": 0.05,  "max_latency": 100.0},
}

# Severity mapping — URLLC issues are always more serious
SEVERITY_BY_SLICE = {
    "URLLC": Severity.CRITICAL,
    "eMBB":  Severity.HIGH,
    "mMTC":  Severity.MEDIUM,
}


class MonitorAgent:
    """
    Observes network metrics and produces structured alerts.

    Maintains a rolling history window per slice so it can detect trends,
    not just instantaneous threshold breaches.
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: Dict[str, Dict[str, deque]] = {}
        self.step = 0

    def _ensure_slice(self, name: str):
        if name not in self.history:
            self.history[name] = {
                "throughput": deque(maxlen=self.window_size),
                "latency": deque(maxlen=self.window_size),
                "allocation": deque(maxlen=self.window_size),
                "utilization": deque(maxlen=self.window_size),
                "queue": deque(maxlen=self.window_size),
            }

    def observe(self, metrics: Dict) -> List[Alert]:
        """
        Process one metrics snapshot and return alerts.

        Args:
            metrics: Output of NetworkSimulator.get_metrics_snapshot()

        Returns:
            List of Alert objects, sorted by severity (critical first).
        """
        self.step = metrics.get("step", self.step + 1)
        alerts: List[Alert] = []

        for name, data in metrics.get("slices", {}).items():
            self._ensure_slice(name)
            h = self.history[name]

            tp = data.get("throughput_mbps", 0.0)
            lat = data.get("latency_ms", 0.0)
            alloc = data.get("allocation", 0.0)
            util = data.get("utilization", 0.0)
            queue = data.get("queue_length", 0)

            h["throughput"].append(tp)
            h["latency"].append(lat)
            h["allocation"].append(alloc)
            h["utilization"].append(util)
            h["queue"].append(queue)

            targets = SLA_TARGETS.get(name, {})
            base_sev = SEVERITY_BY_SLICE.get(name, Severity.MEDIUM)
            snapshot = {
                "throughput_mbps": round(tp, 2),
                "latency_ms": round(lat, 3),
                "allocation": round(alloc, 3),
                "utilization": round(util, 3),
                "queue_length": queue,
            }

            # ── Check 1: latency SLA breach ────────────────────
            max_lat = targets.get("max_latency")
            if max_lat and lat > max_lat:
                alerts.append(Alert(
                    step=self.step, slice_name=name,
                    issue_type=IssueType.LATENCY_HIGH,
                    severity=base_sev,
                    message=f"{name} latency {lat:.2f}ms exceeds {max_lat}ms SLA target",
                    metrics=snapshot,
                ))

            # ── Check 2: throughput SLA breach ─────────────────
            min_tp = targets.get("min_throughput")
            if min_tp and tp < min_tp:
                shortfall_pct = (1 - tp / min_tp) * 100
                # Escalate severity if the shortfall is severe
                sev = Severity.CRITICAL if shortfall_pct > 50 else base_sev
                alerts.append(Alert(
                    step=self.step, slice_name=name,
                    issue_type=IssueType.THROUGHPUT_LOW,
                    severity=sev,
                    message=f"{name} throughput {tp:.1f} Mbps is {shortfall_pct:.0f}% below {min_tp} Mbps target",
                    metrics=snapshot,
                ))

            # ── Check 3: queue buildup (early warning) ─────────
            if queue > 5:
                alerts.append(Alert(
                    step=self.step, slice_name=name,
                    issue_type=IssueType.QUEUE_BUILDUP,
                    severity=Severity.HIGH if queue > 10 else Severity.MEDIUM,
                    message=f"{name} queue at {queue} packets — congestion forming",
                    metrics=snapshot,
                ))

            # ── Check 4: degradation trend ─────────────────────
            if len(h["throughput"]) >= self.window_size:
                series = np.array(h["throughput"])
                slope = np.polyfit(range(len(series)), series, 1)[0]
                # Declining more than 2% of target per step is concerning
                threshold = -(min_tp or 1.0) * 0.02
                if slope < threshold and series[-1] > 0:
                    alerts.append(Alert(
                        step=self.step, slice_name=name,
                        issue_type=IssueType.DEGRADATION_TREND,
                        severity=Severity.MEDIUM,
                        message=f"{name} throughput declining at {slope:.2f} Mbps/step over last {self.window_size} steps",
                        metrics={**snapshot, "trend_slope": round(float(slope), 3)},
                    ))

            # ── Check 5: allocation oscillation (anti-pattern) ─
            if len(h["allocation"]) >= self.window_size:
                variance = float(np.var(h["allocation"]))
                if variance > 0.02:
                    alerts.append(Alert(
                        step=self.step, slice_name=name,
                        issue_type=IssueType.ALLOCATION_OSCILLATION,
                        severity=Severity.LOW,
                        message=f"{name} allocation oscillating (variance {variance:.3f}) — unstable policy",
                        metrics={**snapshot, "alloc_variance": round(variance, 4)},
                    ))

            # ── Check 6: underutilization (wasted resource) ────
            if len(h["utilization"]) >= self.window_size:
                avg_util = float(np.mean(h["utilization"]))
                if avg_util < 0.25 and alloc > 0.25:
                    alerts.append(Alert(
                        step=self.step, slice_name=name,
                        issue_type=IssueType.UNDERUTILIZED,
                        severity=Severity.LOW,
                        message=f"{name} using only {avg_util:.0%} of its {alloc:.0%} allocation — over-provisioned",
                        metrics={**snapshot, "avg_utilization": round(avg_util, 3)},
                    ))

        # Sort: critical first
        order = {
            Severity.CRITICAL: 0, Severity.HIGH: 1,
            Severity.MEDIUM: 2, Severity.LOW: 3, Severity.INFO: 4,
        }
        alerts.sort(key=lambda a: order.get(a.severity, 5))
        return alerts

    def reset(self):
        """Clear history — call when the simulation resets."""
        self.history.clear()
        self.step = 0
