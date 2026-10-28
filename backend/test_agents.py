"""
Test the Phase 5 agentic pipeline end-to-end.

Run:
    python test_agents.py              # rule-based mode (no API key needed)
    python test_agents.py --llm        # with LLM (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
    python test_agents.py --compare    # run all modes and compare
"""

import sys
import argparse
import numpy as np

sys.path.insert(0, ".")

from simulator.network import NetworkSimulator
from agents.pipeline import AgenticPipeline


def run_episode(pipeline, steps: int = 200, label: str = "", verbose: bool = False):
    """Run one episode with the agentic pipeline active."""
    sim = NetworkSimulator(total_bandwidth_mhz=100.0, seed=42)
    sim.reset()
    pipeline.reset()

    total_reward = 0.0
    violations = {"eMBB": 0, "URLLC": 0, "mMTC": 0}

    for i in range(steps):
        metrics = sim.get_metrics_snapshot()

        # Baseline DRL action (equal allocation as a stand-in)
        drl_action = np.array([0.0, 0.0, 0.0])

        # Run the agentic pipeline
        adjusted, trace = pipeline.process(metrics, drl_action, sim.history)

        # Step the simulator with the adjusted action
        obs, reward, done, info = sim.step(adjusted)
        total_reward += reward

        for name, s in info.get("slices", {}).items():
            if not s.get("sla_satisfied", True):
                violations[name] += 1

        # Print interesting traces
        if verbose and trace.diagnosis:
            d = trace.diagnosis
            dec = trace.decision or {}
            print(f"\n  ── Step {trace.step} ──")
            print(f"  Alerts:    {len(trace.alerts)}")
            print(f"  Cause:     {d.get('root_cause')}")
            print(f"  Transient: {d.get('is_transient')} (conf {d.get('confidence', 0):.2f})")
            print(f"  Action:    {dec.get('action_type')}")
            print(f"  Floors:    {dec.get('constraints')}")

    stats = pipeline.get_stats()
    total_violations = sum(violations.values())

    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  Mode:              {stats['mode']}  |  RAG: {stats['rag_enabled']}")
    print(f"  Total reward:      {total_reward:>10.1f}")
    print(f"  Total violations:  {total_violations:>10d}   {violations}")
    print(f"  Alerts raised:     {stats['total_alerts']:>10d}")
    print(f"  Diagnoses run:     {stats['total_diagnoses']:>10d}")
    print(f"  LLM calls:         {stats['total_llm_calls']:>10d}")
    print(f"  Avg latency:       {stats['avg_latency_ms']:>10.2f} ms")
    print(f"  Actions taken:     {stats['action_distribution']}")

    return {
        "reward": total_reward,
        "violations": total_violations,
        "per_slice": violations,
        "stats": stats,
    }


def run_baseline(steps: int = 200):
    """Run without any agentic layer — pure equal allocation."""
    sim = NetworkSimulator(total_bandwidth_mhz=100.0, seed=42)
    sim.reset()

    total_reward = 0.0
    violations = {"eMBB": 0, "URLLC": 0, "mMTC": 0}

    for _ in range(steps):
        action = np.array([0.0, 0.0, 0.0])
        obs, reward, done, info = sim.step(action)
        total_reward += reward
        for name, s in info.get("slices", {}).items():
            if not s.get("sla_satisfied", True):
                violations[name] += 1

    total_violations = sum(violations.values())
    print(f"\n{'=' * 62}")
    print(f"  BASELINE (no agents, equal allocation)")
    print(f"{'=' * 62}")
    print(f"  Total reward:      {total_reward:>10.1f}")
    print(f"  Total violations:  {total_violations:>10d}   {violations}")

    return {"reward": total_reward, "violations": total_violations, "per_slice": violations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true", help="Use LLM reasoning")
    parser.add_argument("--compare", action="store_true", help="Compare all modes")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  Phase 5 — Agentic Pipeline Test")
    print("=" * 62)

    if args.compare:
        results = {}

        results["baseline"] = run_baseline(args.steps)

        p1 = AgenticPipeline(use_llm=False, use_rag=False, diagnose_every=5)
        results["rules_only"] = run_episode(p1, args.steps, "AGENTS: rule-based, no RAG")

        p2 = AgenticPipeline(use_llm=False, use_rag=True, diagnose_every=5)
        results["rules_rag"] = run_episode(p2, args.steps, "AGENTS: rule-based + RAG")

        p3 = AgenticPipeline(use_llm=True, use_rag=True, diagnose_every=5)
        results["llm_rag"] = run_episode(p3, args.steps, "AGENTS: LLM + RAG (full framework)")

        # Summary comparison
        print(f"\n\n{'=' * 62}")
        print(f"  ABLATION SUMMARY ({args.steps} steps)")
        print(f"{'=' * 62}")
        print(f"  {'Configuration':<28} {'Reward':>10} {'Violations':>12}")
        print(f"  {'-' * 52}")
        base_v = results["baseline"]["violations"]
        for key, label in [
            ("baseline", "No agents (baseline)"),
            ("rules_only", "Agents, rules only"),
            ("rules_rag", "Agents + RAG"),
            ("llm_rag", "Agents + RAG + LLM"),
        ]:
            r = results[key]
            delta = ((base_v - r["violations"]) / max(base_v, 1)) * 100
            marker = f"  ({delta:+.0f}%)" if key != "baseline" else ""
            print(f"  {label:<28} {r['reward']:>10.1f} {r['violations']:>12d}{marker}")

    else:
        pipeline = AgenticPipeline(
            use_llm=args.llm,
            use_rag=True,
            diagnose_every=5,
        )
        label = "AGENTS: LLM + RAG" if args.llm else "AGENTS: rule-based + RAG"
        run_episode(pipeline, args.steps, label, verbose=args.verbose)

    print("\n✅ Phase 5 pipeline test complete\n")


if __name__ == "__main__":
    main()
