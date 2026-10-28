"""
Diagnosis Agent — Phase 5

The second stage. Takes alerts from the Monitor, retrieves relevant
standards and diagnostic procedures from the RAG knowledge base, and
reasons about the ROOT CAUSE — not just the symptom.

This is where the research contribution lives. A DRL agent sees
"latency = 1.3ms" and adjusts allocation. This agent determines WHY
latency rose (traffic surge vs channel degradation vs cross-slice
preemption) and whether the cause is transient or persistent — which
changes what the correct response is.
"""

from typing import Dict, List, Optional
import numpy as np

from agents.schemas import Alert, Diagnosis
from agents.llm_client import LLMClient


SYSTEM_PROMPT = """You are a 6G network diagnosis expert embedded in an autonomous slice orchestration system.

Your job: given SLA violation alerts and the current network state, determine the ROOT CAUSE — not just restate the symptom.

You will be given authoritative excerpts from 3GPP standards and network management guides. Ground your reasoning in these excerpts and cite which source informed your conclusion.

Key distinctions you must make:
- Is the cause bandwidth starvation, channel degradation, a traffic surge, or cross-slice preemption?
- Is the condition TRANSIENT (a temporary spike that will resolve) or PERSISTENT (a sustained change requiring a lasting adjustment)? This determines whether a short-term boost or a baseline reallocation is correct.

Respond ONLY with a JSON object in this exact shape:
{
  "root_cause": "<short phrase, e.g. 'bandwidth starvation from eMBB surge'>",
  "reasoning": "<2-3 sentences explaining the causal chain from evidence to conclusion>",
  "is_transient": true or false,
  "confidence": <float 0.0 to 1.0>,
  "recommended_action": "<what should be done, e.g. 'increase URLLC floor to 30% for 20 steps'>",
  "sources_cited": ["<document name>", ...]
}"""


class DiagnosisAgent:
    """
    Performs root-cause analysis on Monitor alerts using RAG-grounded reasoning.

    Falls back to a deterministic rule-based diagnosis when no LLM is
    configured, so the pipeline runs end-to-end without API keys.
    """

    def __init__(self, llm: Optional[LLMClient] = None, use_rag: bool = True):
        self.llm = llm or LLMClient()
        self.rag = None

        if use_rag:
            try:
                from rag.retriever import RAGRetriever
                self.rag = RAGRetriever(use_local=True)
                print(f"📚 RAG connected ({self.rag.status['indexed_chunks']} chunks)")
            except Exception as e:
                print(f"⚠️  RAG unavailable ({e}). Diagnosis will run without grounding.")

    def diagnose(
        self,
        alerts: List[Alert],
        metrics: Dict,
        history: Optional[List[Dict]] = None,
    ) -> Optional[Diagnosis]:
        """
        Analyze alerts and produce a root-cause diagnosis.

        Args:
            alerts: Alerts from the Monitor Agent (sorted, critical first)
            metrics: Current full network snapshot
            history: Recent step history for trend context

        Returns:
            A Diagnosis, or None if there is nothing to diagnose.
        """
        if not alerts:
            return None

        # Focus on the highest-severity alert — that drives the response
        primary = alerts[0]
        step = primary.step

        # ── Retrieve grounding context from RAG ────────────────
        context = ""
        if self.rag:
            try:
                context = self.rag.query_for_diagnosis(
                    slice_name=primary.slice_name,
                    issue_type=primary.issue_type,
                    metrics=primary.metrics,
                )
            except Exception:
                context = ""

        # ── Build the evidence summary ─────────────────────────
        evidence = self._build_evidence(alerts, metrics, history)

        # ── Rule-based fallback diagnosis ──────────────────────
        fallback = self._rule_based_diagnosis(primary, metrics, history)

        # ── LLM reasoning ──────────────────────────────────────
        user_prompt = f"""## Current Alerts
{self._format_alerts(alerts)}

## Network State
{evidence}

## Relevant Standards and Guidelines
{context if context else "(knowledge base unavailable)"}

Diagnose the root cause of the primary alert on the {primary.slice_name} slice."""

        result = self.llm.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback=fallback,
            max_tokens=700,
        )

        return Diagnosis(
            step=step,
            slice_name=primary.slice_name,
            root_cause=result.get("root_cause", fallback["root_cause"]),
            reasoning=result.get("reasoning", fallback["reasoning"]),
            is_transient=bool(result.get("is_transient", fallback["is_transient"])),
            confidence=float(result.get("confidence", fallback["confidence"])),
            recommended_action=result.get("recommended_action", fallback["recommended_action"]),
            sources_cited=result.get("sources_cited", []),
            alerts_considered=[a.to_dict() for a in alerts[:5]],
        )

    # ── Helpers ────────────────────────────────────────────────

    def _format_alerts(self, alerts: List[Alert]) -> str:
        lines = []
        for a in alerts[:6]:
            lines.append(f"- [{a.severity.upper()}] {a.slice_name}: {a.message}")
        return "\n".join(lines)

    def _build_evidence(
        self, alerts: List[Alert], metrics: Dict, history: Optional[List[Dict]]
    ) -> str:
        """Summarize the network state into a compact evidence block."""
        lines = []
        for name, d in metrics.get("slices", {}).items():
            lines.append(
                f"- {name}: allocation {d.get('allocation', 0):.1%}, "
                f"throughput {d.get('throughput_mbps', 0):.1f} Mbps, "
                f"demand {d.get('demand_mbps', 0):.1f} Mbps, "
                f"latency {d.get('latency_ms', 0):.2f}ms, "
                f"utilization {d.get('utilization', 0):.0%}, "
                f"SNR {d.get('avg_snr_db', 0):.1f}dB, "
                f"queue {d.get('queue_length', 0)}"
            )

        # Add trend context from history
        if history and len(history) >= 5:
            recent = history[-10:]
            lines.append("\nRecent trends (last 10 steps):")
            for name in metrics.get("slices", {}).keys():
                allocs = [h.get("slices", {}).get(name, {}).get("allocation", 0) for h in recent]
                demands = [h.get("slices", {}).get(name, {}).get("demand_mbps", 0) for h in recent]
                if allocs and demands:
                    alloc_change = allocs[-1] - allocs[0]
                    demand_change = demands[-1] - demands[0]
                    lines.append(
                        f"- {name}: allocation {alloc_change:+.1%}, "
                        f"demand {demand_change:+.1f} Mbps"
                    )

        return "\n".join(lines)

    def _rule_based_diagnosis(
        self, primary: Alert, metrics: Dict, history: Optional[List[Dict]]
    ) -> Dict:
        """
        Deterministic diagnosis used when no LLM is configured.

        This doubles as the ablation baseline: comparing LLM vs rule-based
        diagnosis quantifies exactly what the LLM reasoning contributes.
        """
        name = primary.slice_name
        slices = metrics.get("slices", {})
        d = slices.get(name, {})

        alloc = d.get("allocation", 0.33)
        demand = d.get("demand_mbps", 0)
        throughput = d.get("throughput_mbps", 0)
        snr = d.get("avg_snr_db", 20)
        queue = d.get("queue_length", 0)

        # Determine transience by checking whether demand recently spiked
        is_transient = True
        if history and len(history) >= 20:
            recent_demand = [
                h.get("slices", {}).get(name, {}).get("demand_mbps", 0)
                for h in history[-20:]
            ]
            if recent_demand:
                early = np.mean(recent_demand[:10])
                late = np.mean(recent_demand[10:])
                # Sustained rise (not a brief spike) = persistent
                is_transient = not (late > early * 1.2 and late > 0)

        # Causal rules, in priority order
        if alloc < 0.20:
            cause = "bandwidth starvation"
            reason = (
                f"{name} holds only {alloc:.0%} of total bandwidth, below the "
                f"minimum recommended floor. Insufficient allocation is the direct "
                f"cause of the observed shortfall."
            )
            action = f"increase {name} allocation to at least 25%"
            conf = 0.85

        elif snr < 12:
            cause = "channel degradation"
            reason = (
                f"Average SNR for {name} is {snr:.1f}dB, well below normal operating "
                f"range. This is a physical-layer condition; reallocation will have "
                f"limited effect until the channel recovers."
            )
            action = "maintain current allocation and wait for channel recovery"
            conf = 0.75

        elif demand > throughput * 1.5 and throughput > 0:
            cause = "traffic surge exceeding allocated capacity"
            reason = (
                f"{name} demand is {demand:.1f} Mbps against {throughput:.1f} Mbps "
                f"delivered — demand exceeds served capacity by "
                f"{(demand/max(throughput,0.01) - 1)*100:.0f}%."
            )
            action = (
                f"temporarily boost {name} allocation" if is_transient
                else f"permanently raise {name} baseline allocation"
            )
            conf = 0.80

        elif queue > 5:
            cause = "queue congestion from processing backlog"
            reason = (
                f"{queue} packets are queued for {name}, indicating that arrivals "
                f"exceed service rate. Latency will continue to grow until the "
                f"backlog clears."
            )
            action = f"increase {name} allocation to drain the queue"
            conf = 0.70

        else:
            cause = "cross-slice resource contention"
            reason = (
                f"No single dominant factor identified for {name}. Total demand "
                f"across slices likely exceeds available capacity, forcing "
                f"proportional degradation."
            )
            action = "apply priority-weighted reallocation across all slices"
            conf = 0.50

        return {
            "root_cause": cause,
            "reasoning": reason,
            "is_transient": is_transient,
            "confidence": conf,
            "recommended_action": action,
            "sources_cited": ["rule-based (no LLM configured)"],
        }
