"use client";

import { useState } from "react";

// ── Types ────────────────────────────────────────────────

export interface AgentAlert {
  step: number;
  slice_name: string;
  issue_type: string;
  severity: string;
  message: string;
  metrics?: Record<string, number>;
}

export interface AgentDiagnosis {
  step: number;
  slice_name: string;
  root_cause: string;
  reasoning: string;
  is_transient: boolean;
  confidence: number;
  recommended_action: string;
  sources_cited?: string[];
}

export interface AgentDecision {
  step: number;
  action_type: string;
  reasoning: string;
  constraints: Record<string, number>;
  duration_steps: number;
}

export interface AgentData {
  alerts: AgentAlert[];
  alert_count: number;
  diagnosis: AgentDiagnosis | null;
  decision: AgentDecision | null;
  latency_ms: number;
  used_llm: boolean;
  llm_calls: number;
}

export interface AgentStats {
  total_steps: number;
  total_alerts: number;
  total_diagnoses: number;
  total_llm_calls: number;
  avg_latency_ms: number;
  mode: string;
  rag_enabled: boolean;
  action_distribution: Record<string, number>;
}

// ── Helpers ──────────────────────────────────────────────

const SEVERITY_STYLE: Record<string, string> = {
  critical: "bg-red-500/15 text-red-400 ring-red-500/30",
  high: "bg-orange-500/15 text-orange-400 ring-orange-500/30",
  medium: "bg-yellow-500/15 text-yellow-400 ring-yellow-500/30",
  low: "bg-slate-500/15 text-slate-400 ring-slate-500/30",
  info: "bg-blue-500/15 text-blue-400 ring-blue-500/30",
};

const SLICE_COLOR: Record<string, string> = {
  eMBB: "#3b82f6",
  URLLC: "#ef4444",
  mMTC: "#22c55e",
};

const ACTION_LABEL: Record<string, string> = {
  no_action: "No action needed",
  drl_unconstrained: "DRL runs free",
  drl_constrained: "DRL with floors",
  temporary_boost: "Temporary boost",
  emergency: "Emergency override",
};

const ACTION_STYLE: Record<string, string> = {
  no_action: "bg-slate-500/15 text-slate-400",
  drl_unconstrained: "bg-blue-500/15 text-blue-400",
  drl_constrained: "bg-purple-500/15 text-purple-400",
  temporary_boost: "bg-amber-500/15 text-amber-400",
  emergency: "bg-red-500/15 text-red-400",
};

function cleanAction(a: string): string {
  // Backend may serialize enums as "ActionType.DRL_CONSTRAINED"
  const bare = a.includes(".") ? a.split(".").pop()! : a;
  return bare.toLowerCase();
}

// ── Sub-components ───────────────────────────────────────

function StageHeader({ n, title, subtitle, active }: {
  n: number; title: string; subtitle: string; active: boolean;
}) {
  return (
    <div className="flex items-center gap-2.5 mb-3">
      <div className={`w-6 h-6 rounded-lg flex items-center justify-center text-[11px] font-bold
        ${active ? "bg-blue-600 text-white" : "bg-white/[0.06] text-slate-600"}`}>
        {n}
      </div>
      <div>
        <h4 className={`text-sm font-semibold ${active ? "text-white" : "text-slate-600"}`}>{title}</h4>
        <p className="text-[10px] text-slate-600">{subtitle}</p>
      </div>
    </div>
  );
}

function AlertRow({ alert }: { alert: AgentAlert }) {
  const style = SEVERITY_STYLE[alert.severity] || SEVERITY_STYLE.low;
  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className={`shrink-0 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wide ring-1 ${style}`}>
        {alert.severity}
      </span>
      <span className="min-w-0 break-words text-[11px] text-slate-400 leading-snug">{alert.message}</span>
    </div>
  );
}

// ── Main Panel ───────────────────────────────────────────

export default function AgentPanel({
  agent, stats, enabled, onToggle,
}: {
  agent: AgentData | null;
  stats: AgentStats | null;
  enabled: boolean;
  onToggle: (v: boolean) => void;
}) {
  const [showAllAlerts, setShowAllAlerts] = useState(false);

  const diagnosis = agent?.diagnosis || null;
  const decision = agent?.decision || null;
  const alerts = agent?.alerts || [];
  const visibleAlerts = showAllAlerts ? alerts : alerts.slice(0, 4);
  const actionKey = decision ? cleanAction(decision.action_type) : "no_action";

  return (
    <section className="rounded-2xl border border-white/[0.06] bg-gradient-to-br from-[#111827] to-[#0f1521] overflow-hidden">

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 sm:px-5 py-4 border-b border-white/[0.06]">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-xl">🧠</span>
          <div className="min-w-0">
            <h2 className="font-bold text-white text-base">Agentic AI Layer</h2>
            <p className="text-[11px] text-slate-500">
              Monitor → Diagnosis → Optimizer, grounded in 3GPP standards
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {stats && enabled && (
            <div className="flex items-center gap-2 text-[10px]">
              <span className={`px-2 py-1 rounded-md ${stats.mode === "llm"
                ? "bg-green-500/15 text-green-400" : "bg-slate-500/15 text-slate-400"}`}>
                {stats.mode === "llm" ? "LLM active" : "Rule-based"}
              </span>
              <span className={`px-2 py-1 rounded-md ${stats.rag_enabled
                ? "bg-green-500/15 text-green-400" : "bg-red-500/15 text-red-400"}`}>
                RAG {stats.rag_enabled ? "on" : "off"}
              </span>
            </div>
          )}

          {/* Toggle */}
          <button
            onClick={() => onToggle(!enabled)}
            aria-label="Toggle agentic layer"
            className={`relative shrink-0 w-12 h-6 rounded-full transition-colors ${
              enabled ? "bg-blue-600" : "bg-white/[0.1]"}`}
          >
            <span className={`absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
              enabled ? "translate-x-6" : "translate-x-0.5"}`} />
          </button>
        </div>
      </div>

      {/* Disabled state */}
      {!enabled && (
        <div className="px-4 sm:px-5 py-8 sm:py-10 text-center">
          <p className="text-slate-500 text-sm mb-1">Agentic layer is off</p>
          <p className="text-slate-600 text-xs max-w-md mx-auto">
            Turn it on to let the agents monitor the network, diagnose SLA violations
            against retrieved standards, and constrain the allocation policy.
          </p>
        </div>
      )}

      {/* Active */}
      {enabled && (
        <>
          {/* Stats strip */}
          {stats && (
            <div className="grid grid-cols-3 sm:grid-cols-5 gap-px bg-white/[0.04]">
              {[
                ["Alerts", stats.total_alerts],
                ["Diagnoses", stats.total_diagnoses],
                ["LLM calls", stats.total_llm_calls],
                ["Avg latency", `${stats.avg_latency_ms} ms`],
                ["Steps", stats.total_steps],
              ].map(([label, val]) => (
                <div key={label as string} className="bg-[#0f1521] px-3 py-2.5">
                  <div className="text-[9px] uppercase tracking-wider text-slate-600">{label}</div>
                  <div className="text-sm font-bold text-white tabular-nums mt-0.5">{val}</div>
                </div>
              ))}
            </div>
          )}

          {/* Three stages */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-px bg-white/[0.04]">

            {/* Stage 1 — Monitor */}
            <div className="min-w-0 bg-[#0f1521] p-4">
              <StageHeader n={1} title="Monitor" subtitle="Detects anomalies"
                active={alerts.length > 0} />
              {alerts.length === 0 ? (
                <p className="text-[11px] text-slate-600 py-3">
                  No anomalies detected. All slices within targets.
                </p>
              ) : (
                <>
                  <div className="divide-y divide-white/[0.04]">
                    {visibleAlerts.map((a, i) => <AlertRow key={i} alert={a} />)}
                  </div>
                  {alerts.length > 4 && (
                    <button onClick={() => setShowAllAlerts(!showAllAlerts)}
                      className="mt-2 text-[10px] text-blue-400 hover:text-blue-300">
                      {showAllAlerts ? "Show less" : `+${alerts.length - 4} more`}
                    </button>
                  )}
                </>
              )}
            </div>

            {/* Stage 2 — Diagnosis */}
            <div className="min-w-0 bg-[#0f1521] p-4">
              <StageHeader n={2} title="Diagnosis" subtitle="Finds root cause via RAG"
                active={!!diagnosis} />
              {!diagnosis ? (
                <p className="text-[11px] text-slate-600 py-3">
                  Waiting for alerts to diagnose.
                </p>
              ) : (
                <div className="space-y-2.5">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: SLICE_COLOR[diagnosis.slice_name] || "#888" }} />
                    <span className="text-xs font-semibold text-white">{diagnosis.slice_name}</span>
                    <span className={`ml-auto px-1.5 py-0.5 rounded text-[9px] font-medium ${
                      diagnosis.is_transient
                        ? "bg-amber-500/15 text-amber-400"
                        : "bg-purple-500/15 text-purple-400"}`}>
                      {diagnosis.is_transient ? "transient" : "persistent"}
                    </span>
                  </div>

                  <div className="rounded-lg bg-white/[0.03] px-3 py-2 border border-white/[0.05]">
                    <div className="text-[9px] uppercase tracking-wider text-slate-600 mb-0.5">
                      Root cause
                    </div>
                    <div className="text-xs text-white font-medium break-words">{diagnosis.root_cause}</div>
                  </div>

                  <p className="text-[11px] text-slate-400 leading-relaxed break-words">
                    {diagnosis.reasoning}
                  </p>

                  <div className="flex items-center gap-2 pt-1">
                    <span className="text-[10px] text-slate-600">Confidence</span>
                    <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                      <div className="h-full rounded-full bg-blue-500 transition-all"
                        style={{ width: `${diagnosis.confidence * 100}%` }} />
                    </div>
                    <span className="text-[10px] text-slate-400 tabular-nums">
                      {(diagnosis.confidence * 100).toFixed(0)}%
                    </span>
                  </div>

                  {diagnosis.sources_cited && diagnosis.sources_cited.length > 0 && (
                    <div className="flex flex-wrap gap-1 pt-1">
                      {diagnosis.sources_cited.slice(0, 3).map((s, i) => (
                        <span key={i}
                          className="px-1.5 py-0.5 rounded bg-white/[0.04] text-[9px] text-slate-500">
                          📄 {s.replace(/\.md$/, "")}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Stage 3 — Optimizer */}
            <div className="min-w-0 bg-[#0f1521] p-4">
              <StageHeader n={3} title="Optimizer" subtitle="Constrains the policy"
                active={!!decision} />
              {!decision ? (
                <p className="text-[11px] text-slate-600 py-3">
                  Waiting for a diagnosis to act on.
                </p>
              ) : (
                <div className="space-y-2.5">
                  <span className={`inline-block px-2 py-1 rounded-md text-[10px] font-semibold ${
                    ACTION_STYLE[actionKey] || ACTION_STYLE.no_action}`}>
                    {ACTION_LABEL[actionKey] || actionKey}
                  </span>

                  <p className="text-[11px] text-slate-400 leading-relaxed break-words">
                    {decision.reasoning}
                  </p>

                  {decision.constraints && Object.keys(decision.constraints).length > 0 && (
                    <div className="rounded-lg bg-white/[0.03] px-3 py-2 border border-white/[0.05]">
                      <div className="text-[9px] uppercase tracking-wider text-slate-600 mb-1.5">
                        Allocation floors applied
                      </div>
                      <div className="space-y-1.5">
                        {Object.entries(decision.constraints).map(([name, val]) => (
                          <div key={name} className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full shrink-0"
                              style={{ backgroundColor: SLICE_COLOR[name] || "#888" }} />
                            <span className="text-[10px] text-slate-500 w-12">{name}</span>
                            <div className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                              <div className="h-full rounded-full transition-all"
                                style={{
                                  width: `${val * 100}%`,
                                  backgroundColor: SLICE_COLOR[name] || "#888",
                                }} />
                            </div>
                            <span className="text-[10px] text-white tabular-nums w-10 text-right">
                              {(val * 100).toFixed(0)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {decision.duration_steps > 1 && (
                    <p className="text-[10px] text-slate-600">
                      Holds for {decision.duration_steps} steps
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Footer */}
          {agent && (
            <div className="px-4 sm:px-5 py-2.5 border-t border-white/[0.06] flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-slate-600">
              <span>Cycle latency: <span className="text-slate-400 tabular-nums">{agent.latency_ms} ms</span></span>
              <span>Alerts this step: <span className="text-slate-400 tabular-nums">{agent.alert_count}</span></span>
              {agent.llm_calls > 0 && (
                <span>LLM calls this step: <span className="text-slate-400 tabular-nums">{agent.llm_calls}</span></span>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
