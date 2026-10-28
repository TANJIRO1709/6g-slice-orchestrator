"""Pydantic models for API request/response validation."""

from pydantic import BaseModel
from typing import Dict, List, Optional


class SliceMetrics(BaseModel):
    allocation: float
    throughput_mbps: float
    latency_ms: float
    demand_mbps: float
    utilization: float
    sla_satisfied: bool
    violations_total: int
    queue_length: int
    avg_snr_db: float
    color: str


class SimulationState(BaseModel):
    step: int
    slices: Dict[str, SliceMetrics]
    total_violations: int


class SimulationConfig(BaseModel):
    total_bandwidth_mhz: float = 100.0
    speed: float = 1.0  # simulation speed multiplier
    strategy: str = "drl"  # "drl", "equal", "embb_heavy", "urllc_heavy"


class ActionRequest(BaseModel):
    allocation: Dict[str, float]  # e.g., {"eMBB": 0.5, "URLLC": 0.3, "mMTC": 0.2}


class AgentDecision(BaseModel):
    step: int
    agent: str
    action: str
    reasoning: str
    timestamp: float


class HistoryPoint(BaseModel):
    step: int
    slices: Dict[str, Dict]
    reward: float
