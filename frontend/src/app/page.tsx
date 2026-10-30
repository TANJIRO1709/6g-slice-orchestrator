"use client";

import { useState, useMemo } from "react";
import { useWebSocket, SimState, SliceData } from "@/hooks/useWebSocket";
import AgentPanel from "@/components/AgentPanel";
import { apiBase } from "@/lib/config";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from "recharts";

// ── Constants ────────────────────────────────────────────

const SLICES = {
  eMBB: {
    color: "#3b82f6",
    label: "eMBB",
    icon: "📡",
    fullName: "Enhanced Mobile Broadband",
    description: "High-speed data: streaming, downloads, video calls",
    slaTarget: "≥10 Mbps, ≤15ms latency",
  },
  URLLC: {
    color: "#ef4444",
    label: "URLLC",
    icon: "🚑",
    fullName: "Ultra-Reliable Low-Latency",
    description: "Mission-critical: robotics, remote surgery, autonomous vehicles",
    slaTarget: "≥1 Mbps, ≤1ms latency, 99.999% reliability",
  },
  mMTC: {
    color: "#22c55e",
    label: "mMTC",
    icon: "📱",
    fullName: "Massive Machine-Type Communication",
    description: "IoT sensors: temperature, parking, agriculture monitors",
    slaTarget: "≥0.05 Mbps, ≤100ms latency",
  },
};

const STRATEGIES = [
  { id: "equal", label: "Equal Split", icon: "⚖️", desc: "33% to each slice" },
  { id: "drl", label: "AI (PPO)", icon: "🤖", desc: "Trained DRL agent" },
  { id: "embb_heavy", label: "eMBB Priority", icon: "📡", desc: "Favor throughput" },
  { id: "urllc_heavy", label: "URLLC Priority", icon: "🚑", desc: "Favor low-latency" },
];

// ── Helper Components ────────────────────────────────────

function StatusBadge({ ok, large }: { ok: boolean; large?: boolean }) {
  if (large) {
    return (
      <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-semibold
        ${ok ? "bg-green-500/15 text-green-400 ring-1 ring-green-500/30" : "bg-red-500/15 text-red-400 ring-1 ring-red-500/30"}`}>
        <span className={`w-2 h-2 rounded-full ${ok ? "bg-green-400" : "bg-red-400 animate-pulse-dot"}`} />
        {ok ? "SLA Met" : "SLA Violated"}
      </span>
    );
  }
  return (
    <span className={`w-2 h-2 rounded-full inline-block ${ok ? "bg-green-400" : "bg-red-400 animate-pulse-dot"}`} />
  );
}

function MetricBox({ label, value, unit, subtitle, warn }: {
  label: string; value: string | number; unit?: string; subtitle?: string; warn?: boolean;
}) {
  return (
    <div className={`min-w-0 rounded-lg p-3 transition-colors
      ${warn ? "bg-red-500/10 border border-red-500/20" : "bg-white/[0.03] border border-white/[0.06]"}`}>
      <div className="text-[11px] uppercase tracking-wider text-slate-500 font-medium">{label}</div>
      <div className="flex items-baseline gap-1 mt-1">
        <span className={`text-xl sm:text-2xl font-bold tabular-nums ${warn ? "text-red-400" : "text-white"}`}>
          {typeof value === "number" ? (value < 10 ? value.toFixed(2) : value.toFixed(1)) : value}
        </span>
        {unit && <span className="text-sm text-slate-500">{unit}</span>}
      </div>
      {subtitle && <div className="text-[10px] text-slate-600 mt-0.5 truncate">{subtitle}</div>}
    </div>
  );
}

function ProgressBar({ value, max, color, label }: {
  value: number; max: number; color: string; label?: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between text-[11px] text-slate-500 mb-1">
          <span>{label}</span>
          <span>{pct.toFixed(0)}%</span>
        </div>
      )}
      <div className="w-full h-2 rounded-full bg-white/[0.05] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500 ease-out"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

// ── Slice Card ───────────────────────────────────────────

function SliceCard({ name, data }: { name: string; data: SliceData }) {
  const info = SLICES[name as keyof typeof SLICES];
  if (!info) return null;

  const allocPct = (data.allocation * 100).toFixed(0);

  return (
    <div className={`min-w-0 rounded-2xl p-4 sm:p-5 border transition-all duration-300 hover:border-opacity-50
      bg-gradient-to-br from-[#111827] to-[#0f1521]
      ${data.sla_satisfied ? "border-white/[0.08]" : "border-red-500/30"}`}
      style={{ borderTopColor: data.sla_satisfied ? info.color + "40" : undefined,
               borderTopWidth: data.sla_satisfied ? "2px" : undefined }}>

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-2 mb-4">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-2xl">{info.icon}</span>
          <div className="min-w-0">
            <h3 className="font-bold text-lg text-white">{info.label}</h3>
            <p className="text-[11px] text-slate-500 sm:max-w-[200px]">{info.description}</p>
          </div>
        </div>
        <StatusBadge ok={data.sla_satisfied} large />
      </div>

      {/* Allocation bar */}
      <ProgressBar value={data.allocation} max={1} color={info.color} label={`Bandwidth: ${allocPct}%`} />

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-2 mt-4">
        <MetricBox
          label="Throughput"
          value={data.throughput_mbps}
          unit="Mbps"
          subtitle={`Demand: ${data.demand_mbps.toFixed(1)} Mbps`}
        />
        <MetricBox
          label="Latency"
          value={data.latency_ms}
          unit="ms"
          warn={!data.sla_satisfied && data.latency_ms > 1}
        />
        <MetricBox
          label="Utilization"
          value={(data.utilization * 100).toFixed(0)}
          unit="%"
        />
        <MetricBox
          label="Violations"
          value={data.violations_total}
          warn={data.violations_total > 0}
          subtitle="cumulative"
        />
      </div>

      {/* SLA target reminder */}
      <div className="mt-3 px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
        <span className="text-[10px] text-slate-600">SLA Target: {info.slaTarget}</span>
      </div>
    </div>
  );
}

// ── Allocation Donut ─────────────────────────────────────

function AllocationDonut({ state }: { state: SimState }) {
  const data = Object.entries(state.slices).map(([name, d]) => ({
    name,
    value: Math.round(d.allocation * 100),
    color: SLICES[name as keyof typeof SLICES]?.color || "#888",
  }));

  return (
    <div className="flex items-center justify-center gap-4">
      <div className="w-[120px] h-[120px] shrink-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={35}
              outerRadius={55}
              paddingAngle={3}
              dataKey="value"
              stroke="none"
            >
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="flex flex-col gap-1.5">
        {data.map((d) => (
          <div key={d.name} className="flex items-center gap-2 text-sm">
            <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: d.color }} />
            <span className="text-slate-400 w-14">{d.name}</span>
            <span className="font-bold text-white tabular-nums">{d.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Charts ───────────────────────────────────────────────

const chartTooltipStyle = {
  backgroundColor: "#1e293b",
  border: "1px solid #334155",
  borderRadius: 12,
  fontSize: 12,
  padding: "8px 12px",
};

function ChartCard({ title, subtitle, children }: {
  title: string; subtitle?: string; children: React.ReactNode;
}) {
  return (
    <div className="min-w-0 rounded-2xl p-3 sm:p-5 bg-gradient-to-br from-[#111827] to-[#0f1521] border border-white/[0.06]">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 mb-4">
        <h3 className="font-semibold text-white text-sm">{title}</h3>
        {subtitle && <span className="text-[11px] text-slate-600">{subtitle}</span>}
      </div>
      {children}
    </div>
  );
}

function ThroughputChart({ history }: { history: SimState[] }) {
  const data = useMemo(() =>
    history.slice(-120).map((h) => ({
      step: h.step,
      eMBB: +(h.slices?.eMBB?.throughput_mbps || 0).toFixed(1),
      URLLC: +(h.slices?.URLLC?.throughput_mbps || 0).toFixed(1),
      mMTC: +(h.slices?.mMTC?.throughput_mbps || 0).toFixed(1),
    })), [history]);

  return (
    <ChartCard title="📈 Throughput Over Time" subtitle="Mbps per slice">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis width={36} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={chartTooltipStyle} />
          <Line type="monotone" dataKey="eMBB" stroke="#3b82f6" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="URLLC" stroke="#ef4444" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="mMTC" stroke="#22c55e" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function AllocationChart({ history }: { history: SimState[] }) {
  const data = useMemo(() =>
    history.slice(-120).map((h) => ({
      step: h.step,
      eMBB: +((h.slices?.eMBB?.allocation || 0.33) * 100).toFixed(0),
      URLLC: +((h.slices?.URLLC?.allocation || 0.33) * 100).toFixed(0),
      mMTC: +((h.slices?.mMTC?.allocation || 0.33) * 100).toFixed(0),
    })), [history]);

  return (
    <ChartCard title="📊 Resource Allocation" subtitle="How bandwidth is split (%)">
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis width={36} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} domain={[0, 100]} />
          <Tooltip contentStyle={chartTooltipStyle} />
          <Area type="monotone" dataKey="mMTC" stackId="1" fill="#22c55e" stroke="#22c55e" fillOpacity={0.5} />
          <Area type="monotone" dataKey="URLLC" stackId="1" fill="#ef4444" stroke="#ef4444" fillOpacity={0.5} />
          <Area type="monotone" dataKey="eMBB" stackId="1" fill="#3b82f6" stroke="#3b82f6" fillOpacity={0.5} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function LatencyChart({ history }: { history: SimState[] }) {
  const data = useMemo(() =>
    history.slice(-120).map((h) => ({
      step: h.step,
      eMBB: +(h.slices?.eMBB?.latency_ms || 0).toFixed(2),
      URLLC: +(h.slices?.URLLC?.latency_ms || 0).toFixed(2),
      mMTC: +(h.slices?.mMTC?.latency_ms || 0).toFixed(2),
    })), [history]);

  return (
    <ChartCard title="⏱ Latency Over Time" subtitle="Lower is better (ms)">
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis width={36} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={chartTooltipStyle} />
          <Line type="monotone" dataKey="eMBB" stroke="#3b82f6" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="URLLC" stroke="#ef4444" dot={false} strokeWidth={2} strokeDasharray="5 3" />
          <Line type="monotone" dataKey="mMTC" stroke="#22c55e" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function RewardChart({ history }: { history: SimState[] }) {
  const data = useMemo(() =>
    history.slice(-120).map((h) => ({
      step: h.step,
      reward: +(h.reward || 0).toFixed(2),
    })), [history]);

  return (
    <ChartCard title="🏆 AI Reward Score" subtitle="Higher = better decisions">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} minTickGap={24} />
          <YAxis width={36} tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <Tooltip contentStyle={chartTooltipStyle} />
          <Bar dataKey="reward" radius={[3, 3, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.reward >= 0 ? "#22c55e" : "#ef4444"} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

// ── Main Dashboard ──────────────────────────────────────

export default function Dashboard() {
  const { state, history, connected, send, clearHistory } = useWebSocket();
  const [strategy, setStrategy] = useState("equal");
  const [speed, setSpeed] = useState(1);
  const [agentsOn, setAgentsOn] = useState(false);
  const post = (path: string) =>
    fetch(`${apiBase()}${path}`, { method: "POST" }).catch(() => {});

  const handleAgentsToggle = (v: boolean) => {
    setAgentsOn(v);
    post(`/api/agents/${v}`);
  };

  const isRunning = state?.is_running || false;

  const handleStart = () => post("/api/start");
  const handlePause = () => post("/api/pause");
  const handleReset = () => { post("/api/reset"); clearHistory(); };
  const handleStep = () => post("/api/step");
  const handleStrategy = (s: string) => {
    setStrategy(s);
    post(`/api/strategy/${s}`);
  };
  const handleSpeed = (v: number) => {
    setSpeed(v);
    post(`/api/speed/${v}`);
  };

  // Calculate summary stats
  const totalViolations = state?.total_violations || 0;
  const avgReward = useMemo(() => {
    if (history.length === 0) return 0;
    const recent = history.slice(-50);
    return recent.reduce((s, h) => s + (h.reward || 0), 0) / recent.length;
  }, [history]);

  const allSatisfied = state?.slices
    ? Object.values(state.slices).every((s) => s.sla_satisfied)
    : true;

  return (
    <div className="min-h-screen">
      {/* ── Top Header Bar ─────────────────────────────── */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#0a0e1a]/80 border-b border-white/[0.06]">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 py-3">
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-blue-600/20 flex items-center justify-center text-lg">🛰️</div>
              <div>
                <h1 className="text-base sm:text-lg font-bold text-white leading-tight">6G Slice Orchestrator</h1>
                <p className="text-[11px] text-slate-500 hidden sm:block">AI-Powered Network Resource Management</p>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-4">
              {/* Connection status */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 animate-pulse-dot" : "bg-red-500"}`} />
                <span className="text-xs text-slate-500">{connected ? "Live" : "Offline"}</span>
              </div>

              {/* Step counter */}
              <div className="px-2 sm:px-3 py-1.5 rounded-lg bg-white/[0.05] border border-white/[0.08]">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Step </span>
                <span className="text-sm font-bold text-white tabular-nums">{state?.step || 0}</span>
              </div>

              {/* Overall health */}
              <div className={`px-2 sm:px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap
                ${allSatisfied
                  ? "bg-green-500/10 text-green-400 border border-green-500/20"
                  : "bg-red-500/10 text-red-400 border border-red-500/20 animate-pulse"}`}>
                {allSatisfied ? "✓ All SLAs Met" : "⚠ SLA Violations"}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-3 sm:px-6 py-4 sm:py-6 space-y-4 sm:space-y-6">

        {/* ── Control Panel ──────────────────────────────── */}
        <section className="rounded-2xl p-4 sm:p-5 bg-gradient-to-br from-[#111827] to-[#0f1521] border border-white/[0.06]">
          <div className="flex flex-col lg:flex-row lg:items-center gap-4">

            {/* Play controls */}
            <div className="flex items-center gap-2 [&>button]:flex-1 sm:[&>button]:flex-none">
              <button onClick={isRunning ? handlePause : handleStart}
                className={`justify-center px-5 py-2.5 rounded-xl font-semibold text-sm flex items-center gap-2 transition-all
                  ${isRunning
                    ? "bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 ring-1 ring-yellow-500/30"
                    : "bg-blue-600 text-white hover:bg-blue-700 shadow-lg shadow-blue-600/25"}`}>
                {isRunning ? "⏸ Pause" : "▶ Start"}
              </button>
              <button onClick={handleStep}
                className="px-4 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.1] text-sm text-slate-300 transition-all border border-white/[0.08]">
                Step →
              </button>
              <button onClick={handleReset}
                className="px-4 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.1] text-sm text-slate-300 transition-all border border-white/[0.08]">
                ↺ Reset
              </button>
            </div>

            {/* Divider */}
            <div className="hidden lg:block h-8 w-px bg-white/[0.08]" />

            {/* Strategy selector */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-slate-600 font-medium">Allocation Strategy</span>
              <div className="flex flex-wrap gap-2">
                {STRATEGIES.map((s) => (
                  <button key={s.id} onClick={() => handleStrategy(s.id)}
                    className={`px-3 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1.5
                      ${strategy === s.id
                        ? "bg-blue-600/20 text-blue-400 ring-1 ring-blue-500/40 shadow-lg shadow-blue-600/10"
                        : "bg-white/[0.03] text-slate-400 hover:bg-white/[0.08] border border-white/[0.06]"}`}>
                    <span>{s.icon}</span>
                    <span>{s.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Divider */}
            <div className="hidden lg:block h-8 w-px bg-white/[0.08]" />

            {/* Speed control */}
            <div className="flex flex-col gap-1.5">
              <span className="text-[10px] uppercase tracking-wider text-slate-600 font-medium">Speed</span>
              <div className="flex flex-wrap gap-1.5">
                {[0.5, 1, 2, 5, 10].map((s) => (
                  <button key={s} onClick={() => handleSpeed(s)}
                    className={`w-10 py-1.5 rounded-lg text-xs font-medium transition-all
                      ${speed === s
                        ? "bg-purple-600/20 text-purple-400 ring-1 ring-purple-500/40"
                        : "bg-white/[0.03] text-slate-500 hover:bg-white/[0.08]"}`}>
                    {s}x
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Strategy explanation */}
          <div className="mt-3 px-3 sm:px-4 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04]">
            <span className="text-[11px] text-slate-600">
              {strategy === "drl" && "🤖 AI agent trained via Proximal Policy Optimization — learns optimal allocation through trial and error"}
              {strategy === "equal" && "⚖️ Static baseline — splits bandwidth equally (33% each), ignoring actual demand"}
              {strategy === "embb_heavy" && "📡 Gives most bandwidth to high-throughput users — URLLC/mMTC may suffer"}
              {strategy === "urllc_heavy" && "🚑 Prioritizes low-latency critical traffic — throughput for eMBB may drop"}
            </span>
          </div>
        </section>

        {/* ── Quick Stats Row ────────────────────────────── */}
        {state?.slices && (
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="col-span-2 sm:col-span-1 min-w-0 rounded-xl p-4 bg-gradient-to-br from-[#111827] to-[#0f1521] border border-white/[0.06]">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider">Current Allocation</div>
              <div className="mt-2">
                <AllocationDonut state={state} />
              </div>
            </div>
            <MetricBox label="Total Violations" value={totalViolations} warn={totalViolations > 0} subtitle="since last reset" />
            <MetricBox label="Avg Reward (last 50)" value={avgReward.toFixed(2)} subtitle={avgReward >= 0 ? "positive — good" : "negative — needs improvement"} />
            <MetricBox label="Data Points" value={history.length} subtitle="recorded steps" />
          </section>
        )}

        {/* ── Agentic AI Layer (Phase 5) ────────────────── */}
        <AgentPanel
          agent={(state?.agent as any) || null}
          stats={(state?.agent_stats as any) || null}
          enabled={agentsOn}
          onToggle={handleAgentsToggle}
        />

        {/* ── Slice Cards ────────────────────────────────── */}
        {state?.slices && (
          <section>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Network Slices — Live Status</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {Object.entries(state.slices).map(([name, data]) => (
                <SliceCard key={name} name={name} data={data} />
              ))}
            </div>
          </section>
        )}

        {/* ── Charts Grid ────────────────────────────────── */}
        {history.length > 2 && (
          <section>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Performance Over Time</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ThroughputChart history={history} />
              <AllocationChart history={history} />
              <LatencyChart history={history} />
              <RewardChart history={history} />
            </div>
          </section>
        )}

        {/* ── Empty state ────────────────────────────────── */}
        {history.length <= 2 && (
          <section className="rounded-2xl p-6 sm:p-12 bg-gradient-to-br from-[#111827] to-[#0f1521] border border-white/[0.06] text-center">
            <div className="text-4xl mb-4">🛰️</div>
            <h3 className="text-xl font-bold text-white mb-2">Ready to Simulate</h3>
            <p className="text-slate-500 text-sm max-w-md mx-auto mb-6">
              Click <strong>Start</strong> above to begin the 6G network simulation.
              Watch how different AI strategies manage bandwidth across eMBB, URLLC, and mMTC slices in real time.
            </p>
            <button onClick={handleStart}
              className="px-6 py-3 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-700 shadow-lg shadow-blue-600/25 transition-all">
              ▶ Start Simulation
            </button>
          </section>
        )}

        {/* ── How It Works (always visible at bottom) ───── */}
        <section className="rounded-2xl p-4 sm:p-5 bg-gradient-to-br from-[#111827] to-[#0f1521] border border-white/[0.06]">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">How It Works</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.04]">
              <span className="text-2xl">📡</span>
              <h4 className="font-semibold text-white mt-2 text-sm">Network Simulation</h4>
              <p className="text-xs text-slate-500 mt-1">
                A virtual 6G base station serves 3 types of traffic with limited bandwidth.
                Channel conditions change every step (Rayleigh fading).
              </p>
            </div>
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.04]">
              <span className="text-2xl">🤖</span>
              <h4 className="font-semibold text-white mt-2 text-sm">AI Decision Making</h4>
              <p className="text-xs text-slate-500 mt-1">
                A PPO-trained agent decides how to split bandwidth each step.
                It learned from 100K+ trial-and-error simulations to minimize SLA violations.
              </p>
            </div>
            <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.04]">
              <span className="text-2xl">📊</span>
              <h4 className="font-semibold text-white mt-2 text-sm">Live Monitoring</h4>
              <p className="text-xs text-slate-500 mt-1">
                This dashboard shows real-time metrics via WebSocket.
                Compare AI vs baseline strategies to see the performance difference.
              </p>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center py-4 text-[11px] text-slate-700">
          6G Slice Orchestrator · Research Project · NIT Rourkela · Sujal Kumar Agarwal (123EI0596)
        </footer>
      </main>
    </div>
  );
}
