"""
Agentic Pipeline Orchestrator — Phase 5

Wires the three agents into a closed loop around the DRL policy:

    metrics ──▶ Monitor ──alerts──▶ Diagnosis ──diagnosis──▶ Optimizer
                                         ▲                       │
                                         │                  constraints
                                    RAG context                  │
                                                                 ▼
                                                    DRL action ──┴──▶ adjusted action

Every cycle produces an AgentTrace containing the full reasoning chain,
which is logged for the dashboard and for the evaluation in Phase 7.
"""

import time
from typing import Dict, List, Optional
import numpy as np

from agents.monitor import MonitorAgent
from agents.diagnosis import DiagnosisAgent
from agents.optimizer import OptimizerAgent
from agents.llm_client import LLMClient
from agents.schemas import AgentTrace, Decision, ActionType


class AgenticPipeline:
    """
    Coordinates Monitor → Diagnosis → Optimizer and applies the result
    to the DRL policy's action.

    Args:
        use_llm: If False, forces rule-based reasoning (ablation baseline).
        use_rag: If False, diagnosis runs without knowledge grounding.
        diagnose_every: Run the (expensive) diagnosis stage only every N
                        steps unless a CRITICAL alert fires. Keeps LLM
                        cost and latency manageable during long runs.
    """

    def __init__(
        self,
        use_llm: bool = True,
        use_rag: bool = True,
        diagnose_every: int = 5,
    ):
        self.llm = LLMClient() if use_llm else LLMClient(provider="rules")
        self.monitor = MonitorAgent(window_size=10)
        self.diagnosis_agent = DiagnosisAgent(llm=self.llm, use_rag=use_rag)
        self.optimizer = OptimizerAgent(llm=self.llm)

        self.diagnose_every = diagnose_every
        self.traces: List[AgentTrace] = []
        self.step_count = 0

        # Cache the last decision so we don't re-diagnose every single step
        self._last_decision: Optional[Decision] = None

        self.slice_order = ["eMBB", "URLLC", "mMTC"]

    def process(
        self,
        metrics: Dict,
        drl_action: np.ndarray,
        history: Optional[List[Dict]] = None,
    ) -> tuple:
        """
        Run one full agentic decision cycle.

        Args:
            metrics: Current network snapshot
            drl_action: Raw action from the DRL policy
            history: Recent step history for trend analysis

        Returns:
            (adjusted_action, trace)
        """
        t0 = time.time()
        self.step_count += 1
        step = metrics.get("step", self.step_count)
        calls_before = self.llm.call_count

        # ── Stage 1: Monitor (every step, cheap) ───────────────
        alerts = self.monitor.observe(metrics)

        # ── Decide whether to run the expensive stages ─────────
        has_critical = any(a.severity == "critical" for a in alerts)
        due_for_diagnosis = (self.step_count % self.diagnose_every == 0)
        should_diagnose = bool(alerts) and (has_critical or due_for_diagnosis)

        diagnosis = None
        decision = self._last_decision

        if should_diagnose:
            # ── Stage 2: Diagnosis (RAG + LLM) ─────────────────
            diagnosis = self.diagnosis_agent.diagnose(alerts, metrics, history)

            # ── Stage 3: Optimizer ─────────────────────────────
            decision = self.optimizer.decide(diagnosis, metrics, alerts)
            self._last_decision = decision

        elif not alerts:
            # Healthy — clear any stale constraint
            decision = self.optimizer.decide(None, metrics, [])
            self._last_decision = decision

        # ── Apply the decision to the DRL action ───────────────
        if decision:
            adjusted = OptimizerAgent.apply(drl_action, decision, self.slice_order)
        else:
            adjusted = drl_action

        # ── Record the trace ───────────────────────────────────
        elapsed_ms = (time.time() - t0) * 1000
        trace = AgentTrace(
            step=step,
            alerts=[a.to_dict() for a in alerts],
            diagnosis=diagnosis.to_dict() if diagnosis else None,
            decision=decision.to_dict() if decision else None,
            llm_calls=self.llm.call_count - calls_before,
            latency_ms=round(elapsed_ms, 2),
            used_llm=self.llm.is_llm,
        )
        self.traces.append(trace)
        if len(self.traces) > 500:
            self.traces = self.traces[-300:]

        return adjusted, trace

    def get_recent_traces(self, n: int = 20) -> List[Dict]:
        """Recent traces for the dashboard."""
        return [t.to_dict() for t in self.traces[-n:]]

    def get_stats(self) -> Dict:
        """Aggregate statistics for evaluation."""
        if not self.traces:
            return {
                "total_steps": 0,
                "total_alerts": 0,
                "total_diagnoses": 0,
                "total_llm_calls": 0,
                "avg_latency_ms": 0.0,
                "mode": "llm" if self.llm.is_llm else "rule-based",
                "rag_enabled": self.diagnosis_agent.rag is not None,
                "action_distribution": {},
            }

        total_alerts = sum(len(t.alerts) for t in self.traces)
        total_diag = sum(1 for t in self.traces if t.diagnosis)
        total_calls = sum(t.llm_calls for t in self.traces)
        avg_latency = float(np.mean([t.latency_ms for t in self.traces]))

        # Count actions taken
        action_counts: Dict[str, int] = {}
        for t in self.traces:
            if t.decision:
                at = t.decision.get("action_type", "unknown")
                action_counts[at] = action_counts.get(at, 0) + 1

        return {
            "total_steps": len(self.traces),
            "total_alerts": total_alerts,
            "total_diagnoses": total_diag,
            "total_llm_calls": total_calls,
            "avg_latency_ms": round(avg_latency, 2),
            "mode": "llm" if self.llm.is_llm else "rule-based",
            "rag_enabled": self.diagnosis_agent.rag is not None,
            "action_distribution": action_counts,
        }

    def reset(self):
        self.monitor.reset()
        self.optimizer.reset()
        self.traces = []
        self.step_count = 0
        self._last_decision = None
