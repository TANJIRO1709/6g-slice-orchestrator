"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { wsUrl } from "@/lib/config";


export interface SliceData {
  allocation: number;
  throughput_mbps: number;
  latency_ms: number;
  demand_mbps: number;
  utilization: number;
  sla_satisfied: boolean;
  violations_total: number;
  queue_length: number;
  avg_snr_db: number;
  color: string;
}

export interface SimState {
  step: number;
  slices: Record<string, SliceData>;
  total_violations: number;
  reward?: number;
  strategy?: string;
  is_running?: boolean;
  agents_enabled?: boolean;
  // Phase 5 — agentic layer
  agent?: {
    alerts: any[];
    alert_count: number;
    diagnosis: any | null;
    decision: any | null;
    latency_ms: number;
    used_llm: boolean;
    llm_calls: number;
  } | null;
  agent_stats?: {
    total_steps: number;
    total_alerts: number;
    total_diagnoses: number;
    total_llm_calls: number;
    avg_latency_ms: number;
    mode: string;
    rag_enabled: boolean;
    action_distribution: Record<string, number>;
  } | null;
}

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const [state, setState] = useState<SimState | null>(null);
  const [history, setHistory] = useState<SimState[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const connect = () => {
      try {
        const socket = new WebSocket(wsUrl());
        ws.current = socket;

        socket.onopen = () => {
          setConnected(true);
          if (reconnectTimer.current) {
            clearTimeout(reconnectTimer.current);
            reconnectTimer.current = null;
          }
        };

        socket.onmessage = (event) => {
          try {
            const data: SimState = JSON.parse(event.data);
            setState(data);
            setHistory((prev) => {
              const next = [...prev, data];
              return next.length > 500 ? next.slice(-500) : next;
            });
          } catch {}
        };

        socket.onclose = () => {
          setConnected(false);
          ws.current = null;
          reconnectTimer.current = setTimeout(connect, 2000);
        };

        socket.onerror = () => {
          socket.close();
        };
      } catch {
        reconnectTimer.current = setTimeout(connect, 2000);
      }
    };

    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
  }, []);

  const send = useCallback((msg: Record<string, unknown>) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(msg));
    }
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return { state, history, connected, send, clearHistory };
}
