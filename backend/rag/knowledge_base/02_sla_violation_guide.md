# SLA Violation Diagnosis and Remediation Guide

## Common SLA Violation Patterns and Root Causes

### Pattern 1: eMBB Throughput Degradation
**Symptoms:** Average eMBB user throughput drops below 100 Mbps target.
**Possible Root Causes:**
- Insufficient bandwidth allocation: eMBB slice has less than 30% of total bandwidth during peak hours.
- Channel degradation: Average SNR across eMBB users has dropped due to fading or interference.
- Traffic surge: Sudden increase in number of active eMBB users or per-user demand (e.g., major streaming event).
- Cross-slice interference: URLLC preemption consuming eMBB resources excessively.

**Diagnosis Steps:**
1. Check current eMBB allocation ratio. If below 30%, bandwidth starvation is the primary cause.
2. Check average SNR for eMBB users. If SNR dropped more than 3dB from baseline, channel degradation is contributing.
3. Check eMBB demand versus capacity. If demand exceeds capacity by more than 50%, a traffic surge is occurring.
4. Check if URLLC allocation recently increased. If URLLC grew by more than 10% in the last 10 steps, cross-slice preemption is the cause.

**Recommended Actions:**
- If bandwidth starvation: Increase eMBB allocation by reducing mMTC (preferred) or reducing URLLC shared pool (last resort).
- If channel degradation: This is a physical layer issue; allocation changes have limited impact. Maintain current allocation and wait for channel recovery.
- If traffic surge: Temporarily increase eMBB allocation. If surge appears periodic (time-of-day pattern), the adjustment should be temporary (30-50 steps) then restored.
- If cross-slice preemption: Set a minimum eMBB allocation floor of 25% that URLLC preemption cannot breach.

### Pattern 2: URLLC Latency Violation
**Symptoms:** URLLC average latency exceeds 1ms target. This is a CRITICAL violation.
**Possible Root Causes:**
- Insufficient URLLC bandwidth: The slice has too little bandwidth to serve all URLLC users within latency requirements.
- Queue buildup: URLLC packet queue is growing, indicating processing delay.
- High URLLC demand: More URLLC devices than expected are transmitting simultaneously (e.g., factory alarm event).
- Resource contention: Other slices are consuming resources that URLLC needs.

**Diagnosis Steps:**
1. Check URLLC allocation ratio. If below 20%, bandwidth starvation is very likely the cause.
2. Check URLLC queue length. If queue > 5, packets are backing up and latency will continue to worsen.
3. Check if URLLC demand spiked. Compare current demand to the 10-step moving average.
4. Check utilization of other slices. If eMBB or mMTC utilization is above 90%, they may be consuming shared resources.

**Recommended Actions — URLLC violations require IMMEDIATE response:**
- Increase URLLC allocation to at least 30% by taking from mMTC first, then eMBB.
- Set URLLC allocation floor: The optimizer should never allow URLLC below 20% regardless of other demands.
- If demand spike is temporary (alarm event), increase allocation for 20 steps then gradually restore.
- If demand spike is sustained (new URLLC devices added), the allocation increase should be permanent.

### Pattern 3: mMTC Connection Failures
**Symptoms:** mMTC throughput drops below 0.1 Mbps or many devices cannot transmit.
**Possible Root Causes:**
- Bandwidth starvation: mMTC has been reduced to near-zero allocation due to eMBB/URLLC demands.
- Device collision: Too many mMTC devices attempting to transmit simultaneously.
- Low priority scheduling: mMTC is being consistently deprioritized.

**Diagnosis Steps:**
1. Check mMTC allocation ratio. If below 5%, the slice is being starved.
2. Check number of active mMTC devices versus allocated capacity.
3. Check if mMTC allocation has been continuously decreasing over the last 20 steps.

**Recommended Actions:**
- Ensure mMTC always has at least 5% minimum allocation.
- mMTC violations are lower severity — acceptable to defer if URLLC or eMBB are also under pressure.
- If mMTC starvation persists for more than 50 consecutive steps, it should be escalated.

## Violation Severity Classification

| Slice | Severity | Response Time |
|-------|----------|---------------|
| URLLC latency > 1ms | CRITICAL | Immediate (next step) |
| URLLC latency > 0.8ms | WARNING | Within 5 steps |
| eMBB throughput < 100 Mbps | HIGH | Within 10 steps |
| eMBB throughput < 80 Mbps | CRITICAL | Within 5 steps |
| mMTC throughput < 0.1 Mbps | MEDIUM | Within 20 steps |
| mMTC allocation < 5% | HIGH | Within 10 steps |
| Any slice utilization > 95% | WARNING | Monitor for 5 steps |
| Any slice queue length > 10 | HIGH | Within 5 steps |

## Traffic Pattern Recognition

### Time-of-Day Patterns
- eMBB traffic typically follows a sinusoidal pattern: peaks during evening hours (streaming), dips during early morning.
- URLLC traffic in industrial settings peaks during working hours, drops during nights and weekends.
- mMTC sensor traffic may have periodic bursts (e.g., all sensors reporting at the same interval).

### Event-Driven Spikes
- eMBB: Major sports events, concerts, or emergencies cause sudden spikes lasting 1-3 hours.
- URLLC: Factory alarms, emergency braking events, or surgical procedures cause short intense bursts lasting seconds to minutes.
- mMTC: Environmental events (storms, earthquakes) may cause all sensors to report simultaneously.

### Identifying Temporary vs Permanent Changes
- If demand increase correlates with time-of-day and has occurred in previous cycles, it is TEMPORARY — use short-term allocation adjustment.
- If demand increase is monotonic and does not correlate with periodic patterns, it is likely PERMANENT — adjust baseline allocation.
- Rule of thumb: If a demand change persists for more than 100 steps without reverting, treat it as permanent.
