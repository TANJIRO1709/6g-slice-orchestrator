# Resource Allocation Best Practices for 6G Network Slicing

## General Principles

### Principle 1: Demand-Proportional Allocation
Allocate resources proportional to current demand, not fixed ratios. A slice with 70% of the total demand should receive approximately 70% of resources, adjusted for priority weights. Static equal allocation (33% each) is the worst-case baseline and should always be outperformed by any intelligent strategy.

### Principle 2: Priority-Weighted Allocation
When demand exceeds supply, apply priority weights to determine which slice to reduce:
- URLLC weight: 3.0 (highest priority — never reduce below minimum)
- eMBB weight: 1.5 (medium priority — reduce proportionally)
- mMTC weight: 1.0 (lowest priority — reduce first)

Formula: allocated_ratio = (demand × priority_weight) / sum(all_demands × all_weights)

### Principle 3: Minimum Allocation Floors
Never reduce any slice below its minimum guaranteed allocation:
- eMBB minimum: 20% of total bandwidth
- URLLC minimum: 15% of total bandwidth
- mMTC minimum: 5% of total bandwidth
- Shared pool: remaining 60% allocated dynamically

### Principle 4: Stability Over Optimality
Avoid rapid oscillation of allocation ratios. Large step-to-step changes in allocation cause transient performance degradation as users adapt to new bandwidth. Limit allocation changes to a maximum of 10% shift per step unless responding to a critical URLLC violation. Smooth transitions are preferred: if a 20% shift is needed, apply it over 3-4 steps as 5-7% increments.

### Principle 5: Anticipatory Allocation
If traffic patterns are predictable (time-of-day cycles, periodic mMTC bursts), adjust allocation BEFORE the demand arrives, not after. Pre-allocation reduces latency spikes during traffic ramps and prevents transient SLA violations.

## Specific Scenarios and Recommended Strategies

### Scenario: Normal Steady-State Operation
- All slices operating within SLA targets.
- Recommended: Maintain current allocation with minor demand-proportional adjustments.
- Do not over-optimize — stable good performance is better than constantly chasing optimal.

### Scenario: eMBB Traffic Surge (e.g., evening streaming peak)
- eMBB demand doubles or triples within a short period.
- Recommended: Increase eMBB allocation by taking from mMTC first (mMTC is latency-tolerant).
- If mMTC is already at minimum (5%), take from the shared pool only.
- Never reduce URLLC allocation to serve eMBB surges.
- Duration: Apply for the expected surge duration (typically 50-200 steps for time-of-day patterns).

### Scenario: URLLC Emergency Demand
- URLLC demand spikes due to emergency event (factory alarm, autonomous vehicle emergency stop).
- Recommended: IMMEDIATELY increase URLLC allocation to at least 30%.
- Take resources from mMTC first, then eMBB.
- This is the only scenario where aggressive reallocation is justified over stability.
- Duration: Short-term (10-30 steps). URLLC emergencies are typically brief.

### Scenario: mMTC Burst (all sensors reporting simultaneously)
- mMTC device count suddenly activates (e.g., environmental event triggers all sensors).
- Recommended: Slightly increase mMTC allocation by 5-10%.
- mMTC bursts are typically self-resolving as devices return to sleep after reporting.
- Do not over-allocate to mMTC — it is the lowest priority slice.

### Scenario: Channel Degradation (low SNR across all slices)
- Overall channel quality drops due to environmental factors (rain, interference).
- Recommended: Do NOT aggressively reallocate between slices. The problem is physical, not allocation-related.
- Instead, ensure URLLC maintains its allocation (it is most sensitive to channel quality).
- Accept temporary eMBB throughput reduction until channel recovers.
- Channel recovery is typically automatic within 10-50 steps.

### Scenario: All Slices Under Pressure
- Total demand exceeds total capacity across all slices simultaneously.
- Recommended: Apply strict priority ordering: URLLC > eMBB > mMTC.
- Reduce mMTC to minimum (5%), allocate URLLC its guaranteed 15% plus whatever it needs from the shared pool, give remainder to eMBB.
- This is a capacity limitation — allocation can mitigate but not solve it. Report the overload condition.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Ping-Pong Allocation
Rapidly switching allocation back and forth between two slices every step. This happens when the optimizer over-reacts to each individual step's metrics. Solution: Use smoothed metrics (5-step moving average) for allocation decisions.

### Anti-Pattern 2: Starvation
Reducing one slice to zero or near-zero allocation for extended periods. Even low-priority mMTC should maintain minimum 5% allocation. Starvation causes device timeouts, connection drops, and recovery storms when resources are restored.

### Anti-Pattern 3: Ignoring URLLC
Treating URLLC like other slices during allocation. URLLC has uniquely strict requirements (1ms, 99.999%) and must always be treated as the highest priority. A single URLLC violation is more serious than 100 mMTC violations.

### Anti-Pattern 4: Over-Allocation to URLLC
Giving URLLC 50%+ of resources when it only needs 15-20%. URLLC has low throughput demand — giving it excess bandwidth wastes resources that eMBB could use productively. Allocate based on URLLC demand, not fear of violations.

## Key Metrics for Allocation Decisions

| Metric | Where to Check | Action Trigger |
|--------|---------------|----------------|
| Throughput per slice | Slice state | Below SLA target |
| Latency per slice | Slice state | Above SLA target |
| Utilization per slice | Slice state | Above 90% = overloaded, below 30% = over-provisioned |
| Queue length | Slice state | Above 5 = congestion forming |
| SNR trend | Channel state | Dropping 3dB+ = channel degradation |
| Demand vs capacity ratio | Traffic + allocation | Above 1.0 = demand exceeds allocated capacity |
| Allocation stability | Last 10 steps | Variance > 0.05 = allocation is oscillating |
