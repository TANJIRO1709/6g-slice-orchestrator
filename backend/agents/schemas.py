"""
Shared data structures for inter-agent communication.

The three agents pass structured messages to each other:
    Monitor  → Alert       → Diagnosis
    Diagnosis → Diagnosis  → Optimizer
    Optimizer → Decision   → DRL policy

Every message is JSON-serializable so the full reasoning trace can be
logged, displayed on the dashboard, and audited after the fact.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
from enum import Enum
import time


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(str, Enum):
    SLA_VIOLATION = "sla_violation"
    LATENCY_HIGH = "latency_high"
    THROUGHPUT_LOW = "throughput_low"
    QUEUE_BUILDUP = "queue_buildup"
    DEGRADATION_TREND = "degradation_trend"
    UNDERUTILIZED = "underutilized"
    ALLOCATION_OSCILLATION = "allocation_oscillation"


class ActionType(str, Enum):
    """What kind of intervention the optimizer chose."""
    NO_ACTION = "no_action"                  # everything healthy
    DRL_UNCONSTRAINED = "drl_unconstrained"  # let DRL decide freely
    DRL_CONSTRAINED = "drl_constrained"      # DRL with allocation floors
    EMERGENCY_REALLOCATION = "emergency"     # direct override, bypass DRL
    TEMPORARY_BOOST = "temporary_boost"      # time-limited allocation change


@dataclass
class Alert:
    """Emitted by the Monitor Agent when something looks wrong."""
    step: int
    slice_name: str
    issue_type: str
    severity: str
    message: str
    metrics: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Diagnosis:
    """Produced by the Diagnosis Agent after RAG-grounded reasoning."""
    step: int
    slice_name: str
    root_cause: str
    reasoning: str
    is_transient: bool          # temporary spike vs persistent change
    confidence: float           # 0.0 - 1.0
    recommended_action: str
    sources_cited: List[str] = field(default_factory=list)
    alerts_considered: List[Dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Decision:
    """Produced by the Optimizer Agent. Bounds the DRL policy's action."""
    step: int
    action_type: str
    reasoning: str
    # Allocation floors the DRL policy must respect (slice -> min ratio)
    constraints: Dict[str, float] = field(default_factory=dict)
    # If action_type is EMERGENCY_REALLOCATION, this is the direct allocation
    direct_allocation: Optional[Dict[str, float]] = None
    # How many steps this decision remains in effect
    duration_steps: int = 1
    diagnosis_used: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentTrace:
    """
    Full reasoning trace for one decision cycle.
    This is what gets displayed on the dashboard and logged for the paper.
    """
    step: int
    alerts: List[Dict] = field(default_factory=list)
    diagnosis: Optional[Dict] = None
    decision: Optional[Dict] = None
    llm_calls: int = 0
    latency_ms: float = 0.0
    used_llm: bool = False       # False = fell back to rule-based
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)
