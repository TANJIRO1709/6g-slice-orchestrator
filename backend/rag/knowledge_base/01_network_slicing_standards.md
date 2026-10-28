# 3GPP Network Slicing Standards and Requirements

## Overview of Network Slicing (3GPP TR 28.801)

Network slicing allows a single physical 6G network infrastructure to be partitioned into multiple virtual networks called slices. Each slice is an isolated, end-to-end logical network tailored to fulfill specific service requirements. Network slicing is a fundamental feature of 5G and 6G architectures, enabling operators to serve diverse use cases with differing performance demands on a shared infrastructure.

## Slice Types and Service Categories

### eMBB (Enhanced Mobile Broadband) — SST Value 1
- Primary use case: High-throughput data services including video streaming, augmented reality, virtual reality, cloud gaming, and large file downloads.
- Minimum throughput requirement: 100 Mbps per user in dense urban deployments, 50 Mbps in suburban.
- Maximum acceptable latency: 10 milliseconds for interactive applications, 50 milliseconds for streaming.
- Reliability target: 99.9% packet delivery rate.
- Typical user density: 10-50 users per cell in urban macro deployment.
- Traffic pattern: Bursty and asymmetric, with downlink-heavy traffic. Peak-to-average ratio can exceed 10:1.
- Resource priority: Lower than URLLC but higher than mMTC. Can be preempted when URLLC demands spike.
- Key performance indicator: Average user throughput and cell edge throughput.

### URLLC (Ultra-Reliable Low-Latency Communications) — SST Value 2
- Primary use case: Mission-critical applications including remote surgery, autonomous vehicle control, industrial robotics, drone control, and emergency services.
- Minimum throughput requirement: 1 Mbps (small control packets but must arrive on time).
- Maximum acceptable latency: 1 millisecond end-to-end for control plane, 0.5 milliseconds for user plane in extreme cases.
- Reliability target: 99.999% packet delivery rate (five nines). This means no more than 1 packet lost per 100,000 transmitted.
- Typical user density: 5-20 critical devices per cell.
- Traffic pattern: Small, periodic, highly deterministic packets. Typical packet size is 32-256 bytes.
- Resource priority: HIGHEST. URLLC traffic must always be served first. Any SLA violation for URLLC is considered a critical incident.
- Key performance indicator: End-to-end latency at the 99.999th percentile, not average latency.
- Special requirements: Preemptive scheduling — URLLC packets can interrupt ongoing eMBB transmissions. Dedicated mini-slots for ultra-low latency.

### mMTC (Massive Machine-Type Communications) — SST Value 3
- Primary use case: Internet of Things sensors, smart city infrastructure, agricultural monitoring, environmental sensors, smart meters, and wearable health devices.
- Minimum throughput requirement: 0.1 Mbps per device (very small data transmissions).
- Maximum acceptable latency: 100 milliseconds (latency-tolerant applications).
- Reliability target: 99% packet delivery rate.
- Typical device density: Up to 1,000,000 devices per square kilometer.
- Traffic pattern: Sporadic and infrequent. Each device may transmit only once every few minutes or hours. At any given moment, only 1-5% of connected devices are actively transmitting.
- Resource priority: LOWEST. mMTC traffic should be served opportunistically when eMBB and URLLC have spare capacity.
- Key performance indicator: Connection density and battery life of devices.
- Special requirements: Energy efficiency is critical since most mMTC devices run on batteries for years.

## Slice Isolation Requirements

According to 3GPP TS 23.501, network slices must maintain isolation in multiple dimensions:
- Performance isolation: SLA violations in one slice must not affect the performance of other slices.
- Security isolation: Traffic from one slice must not be visible to or accessible from another slice.
- Resource isolation: Resources allocated to one slice should not be silently consumed by another. However, dynamic resource sharing across slices is permitted when it does not violate SLA constraints.

## Resource Management Principles

### Dynamic Resource Allocation
- Resources (bandwidth, power, compute) should be dynamically allocated based on current demand rather than statically provisioned.
- Allocation decisions should be made at sub-second timescales to respond to fast-changing traffic patterns.
- Over-provisioning wastes resources; under-provisioning causes SLA violations. The optimal strategy balances utilization against SLA compliance risk.

### Priority-Based Preemption
- When total demand exceeds available resources, lower-priority slices should be preempted first.
- Priority order: URLLC > mMTC > eMBB for latency-sensitive decisions. URLLC > eMBB > mMTC for general resource allocation.
- Preemption should be proportional: reduce the lowest-priority slice's allocation gradually rather than cutting it entirely.

### Minimum Guaranteed Resources
- Each slice should have a minimum guaranteed resource allocation that cannot be preempted, even under extreme load.
- Typical minimums: eMBB — 20% of total bandwidth, URLLC — 15% (small but guaranteed), mMTC — 5%.
- The remaining 60% is a shared pool allocated dynamically based on demand.
