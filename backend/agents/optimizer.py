"""
Optimizer Agent — Phase 5

The third stage. Takes the Diagnosis and decides HOW to act:
    - Let the DRL policy run unconstrained (healthy network)
    - Constrain the DRL policy with allocation floors (targeted fix)
    - Override the DRL policy entirely (URLLC emergency)

This is the mechanism that makes the framework "context-aware constrained
DRL". Instead of a fixed reward function, the constraints applied to the
policy are derived at decision time from standards retrieved by RAG and
reasoning produced by the Diagnosis Agent.
"""

from typing import Dict, List, Optional
import numpy as np

from agents.schemas import Diagnosis, Decision, ActionType, Alert
from agents.llm_client import LLMClient


# From 3GPP guidance in the knowledge base
MIN_FLOORS = {"eMBB": 0.20, "URLLC": 0.15, "mMTC": 0.05}
PRIORITY = {"URLLC": 3.0, "eMBB": 1.5, "mMTC": 1.0}


SYSTEM_PROMPT = """You are a 6G resource optimization controller. You decide how to constrain a deep reinforcement learning policy that allocates bandwidth across three network slices (eMBB, URLLC, mMTC).

You receive a root-cause diagnosis and the current network state. Your job is to choose an intervention and specify numeric constraints.

Available action types:
- "no_action": network is healthy, no intervention needed
- "drl_unconstrained": let the DRL policy allocate freely
- "drl_constrained": DRL allocates but must respect minimum floors you specify
- "emergency": override DRL entirely with a direct allocation (URLLC critical only)
- "temporary_boost": time-limited constraint that expires after N steps

Hard rules you must never violate:
- URLLC allocation must never drop below 0.15
- eMBB allocation must never drop below 0.20
- mMTC allocation must never drop below 0.05
- Constraint floors must sum to at most 0.85, leaving headroom for the DRL policy
- Prefer stability: only use "emergency" for critical URLLC violations

Respond ONLY with a JSON object in this exact shape:
{
  "action_type": "<one of the five types above>",
  "reasoning": "<1-2 sentences on why this intervention fits the diagnosis>",
  "constraints": {"eMBB": <float>, "URLLC": <float>, "mMTC": <float>},
  "duration_steps": <int, how long this stays in effect>
}"""


class OptimizerAgent:
    """
    Converts a diagnosis into an actionable constraint set for the DRL policy.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient()
        # Active constraint with remaining lifetime
        self._active: Optional[Decision] = None
        self._remaining = 0

    def decide(
        self,
        diagnosis: Optional[Diagnosis],
        metrics: Dict,
        alerts: Optional[List[Alert]] = None,
    ) -> Decision:
        """
        Choose an intervention based on the diagnosis.

        Returns a Decision that the runtime applies to the DRL action.
        """
        step = metrics.get("step", 0)

        # ── An active constraint is still in effect ────────────
        if self._active and self._remaining > 0:
            self._remaining -= 1
            held = Decision(
                step=step,
                action_type=self._active.action_type,
                reasoning=f"{self._active.reasoning} (holding, {self._remaining} steps remain)",
                constraints=self._active.constraints,
                direct_allocation=self._active.direct_allocation,
                duration_steps=self._remaining,
                diagnosis_used=self._active.diagnosis_used,
            )
            return held

        # ── Nothing wrong — let DRL run free ───────────────────
        if diagnosis is None:
            self._active = None
            return Decision(
                step=step,
                action_type=ActionType.NO_ACTION,
                reasoning="All slices within SLA targets. No intervention required.",
                constraints={},
                duration_steps=1,
            )

        # ── Build the fallback decision ────────────────────────
        fallback = self._rule_based_decision(diagnosis, metrics)

        # ── LLM reasoning ──────────────────────────────────────
        user_prompt = f"""## Diagnosis
Slice: {diagnosis.slice_name}
Root cause: {diagnosis.root_cause}
Reasoning: {diagnosis.reasoning}
Transient: {diagnosis.is_transient}
Confidence: {diagnosis.confidence:.2f}
Recommended: {diagnosis.recommended_action}

## Current Allocation and State
{self._format_state(metrics)}

Choose an intervention and specify the constraint floors."""

        result = self.llm.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            fallback=fallback,
            max_tokens=500,
        )

        # ── Validate and sanitize the LLM's constraints ────────
        constraints = self._sanitize(result.get("constraints", fallback["constraints"]))
        action_type = result.get("action_type", fallback["action_type"])
        duration = int(result.get("duration_steps", fallback["duration_steps"]))
        duration = max(1, min(duration, 100))  # cap at 100 steps

        decision = Decision(
            step=step,
            action_type=action_type,
            reasoning=result.get("reasoning", fallback["reasoning"]),
            constraints=constraints,
            duration_steps=duration,
            diagnosis_used=diagnosis.to_dict(),
        )

        # Latch the decision for its duration
        if duration > 1:
            self._active = decision
            self._remaining = duration - 1
        else:
            self._active = None
            self._remaining = 0

        return decision

    # ── Applying the decision to a DRL action ──────────────────

    @staticmethod
    def apply(action: np.ndarray, decision: Decision, slice_order: List[str]) -> np.ndarray:
        """
        Apply the decision's constraints to a raw DRL action vector.

        The DRL action is a logit vector that gets softmaxed into an
        allocation. We softmax it, enforce the floors, renormalize, then
        convert back to logits so the downstream simulator behaves the same.

        Args:
            action: Raw DRL action (logits), shape (3,)
            decision: The Optimizer's decision
            slice_order: Slice names in the action vector's order

        Returns:
            Adjusted action vector.
        """
        if decision.action_type == ActionType.NO_ACTION:
            return action
        if decision.action_type == ActionType.DRL_UNCONSTRAINED:
            return action

        # Emergency: direct allocation override
        if decision.action_type == ActionType.EMERGENCY_REALLOCATION and decision.direct_allocation:
            alloc = np.array([
                decision.direct_allocation.get(n, 1.0 / len(slice_order))
                for n in slice_order
            ])
            alloc = alloc / alloc.sum()
            return np.log(alloc + 1e-8)

        if not decision.constraints:
            return action

        # Softmax the raw action into an allocation
        exp_a = np.exp(action - np.max(action))
        alloc = exp_a / exp_a.sum()

        # Enforce floors
        floors = np.array([decision.constraints.get(n, 0.0) for n in slice_order])
        total_floor = floors.sum()
        if total_floor >= 1.0:
            floors = floors / total_floor * 0.95
            total_floor = floors.sum()

        # Distribute the remaining headroom proportional to the DRL's preference
        headroom = 1.0 - total_floor
        above_floor = np.maximum(alloc - floors, 0)
        if above_floor.sum() > 1e-8:
            distributed = above_floor / above_floor.sum() * headroom
        else:
            distributed = np.ones_like(floors) * headroom / len(floors)

        final = floors + distributed
        final = final / final.sum()

        return np.log(final + 1e-8)

    # ── Helpers ────────────────────────────────────────────────

    def _format_state(self, metrics: Dict) -> str:
        lines = []
        for name, d in metrics.get("slices", {}).items():
            status = "OK" if d.get("sla_satisfied", True) else "VIOLATED"
            lines.append(
                f"- {name}: {d.get('allocation', 0):.1%} allocated, "
                f"{d.get('throughput_mbps', 0):.1f} Mbps, "
                f"{d.get('latency_ms', 0):.2f}ms latency, SLA {status}"
            )
        return "\n".join(lines)

    def _sanitize(self, constraints: Dict) -> Dict[str, float]:
        """Clamp LLM-provided constraints to safe, valid values."""
        clean = {}
        for name, floor in MIN_FLOORS.items():
            raw = constraints.get(name, floor)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = floor
            # Never below the hard minimum, never above 0.7
            clean[name] = max(floor, min(val, 0.70))

        # Ensure floors leave headroom for the DRL policy
        total = sum(clean.values())
        if total > 0.85:
            scale = 0.85 / total
            clean = {k: max(MIN_FLOORS[k], v * scale) for k, v in clean.items()}

        return {k: round(v, 3) for k, v in clean.items()}

    def _rule_based_decision(self, diagnosis: Diagnosis, metrics: Dict) -> Dict:
        """
        Deterministic decision logic — the ablation baseline.
        Encodes the allocation best practices from the knowledge base.
        """
        name = diagnosis.slice_name
        slices = metrics.get("slices", {})
        constraints = dict(MIN_FLOORS)

        # URLLC violations are always critical → aggressive response
        if name == "URLLC":
            constraints["URLLC"] = 0.30
            constraints["mMTC"] = 0.05   # take from mMTC first
            constraints["eMBB"] = 0.20
            action = ActionType.EMERGENCY_REALLOCATION if diagnosis.confidence > 0.7 else ActionType.DRL_CONSTRAINED
            reason = (
                f"URLLC violation is critical ({diagnosis.root_cause}). Raising URLLC "
                f"floor to 30%, reducing mMTC to its 5% minimum per priority ordering."
            )
            duration = 20 if diagnosis.is_transient else 60

        elif name == "eMBB":
            constraints["eMBB"] = 0.45
            constraints["mMTC"] = 0.05   # take from mMTC, not URLLC
            constraints["URLLC"] = 0.15
            action = ActionType.TEMPORARY_BOOST if diagnosis.is_transient else ActionType.DRL_CONSTRAINED
            reason = (
                f"eMBB shortfall from {diagnosis.root_cause}. Raising eMBB floor to 45% "
                f"by reducing mMTC, preserving URLLC guarantee."
            )
            duration = 30 if diagnosis.is_transient else 80

        else:  # mMTC
            constraints["mMTC"] = 0.15
            constraints["eMBB"] = 0.20
            constraints["URLLC"] = 0.15
            action = ActionType.DRL_CONSTRAINED
            reason = (
                f"mMTC issue ({diagnosis.root_cause}). Modest floor increase to 15%; "
                f"mMTC is lowest priority so intervention is conservative."
            )
            duration = 15

        # Channel degradation is physical — don't thrash allocations
        if "channel" in diagnosis.root_cause.lower():
            action = ActionType.DRL_UNCONSTRAINED
            reason = (
                "Root cause is channel degradation, a physical-layer condition. "
                "Reallocation cannot fix it; letting DRL operate normally avoids "
                "destabilizing the network while the channel recovers."
            )
            constraints = dict(MIN_FLOORS)
            duration = 10

        return {
            "action_type": action,
            "reasoning": reason,
            "constraints": constraints,
            "duration_steps": duration,
        }

    def reset(self):
        self._active = None
        self._remaining = 0
