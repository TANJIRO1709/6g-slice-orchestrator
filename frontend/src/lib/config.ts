/**
 * Resolves where the backend lives.
 *
 * Local development: Next runs on :3000, FastAPI on :8000, so
 * requests are sent explicitly to :8000.
 *
 * Deployed on Render: FastAPI serves the compiled dashboard itself,
 * so the backend is the same origin and fetches use relative paths.
 *
 * Deployed anywhere else (e.g. Vercel): requests go to the Render
 * backend at RENDER_BACKEND.
 *
 * Either can be overridden with NEXT_PUBLIC_API_URL /
 * NEXT_PUBLIC_WS_URL when the two are deployed separately.
 */

const RENDER_BACKEND = "https://sixg-slice-orchestrator.onrender.com";

function isLocalHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1";
}

export function apiBase(): string {
  const override = process.env.NEXT_PUBLIC_API_URL;
  if (override) return override.replace(/\/$/, "");

  if (typeof window === "undefined") return "http://localhost:8000";

  const { protocol, hostname } = window.location;
  if (isLocalHost(hostname)) return `${protocol}//${hostname}:8000`;

  if (window.location.host === new URL(RENDER_BACKEND).host) return ""; // same origin

  return RENDER_BACKEND;
}

export function wsUrl(): string {
  const override = process.env.NEXT_PUBLIC_WS_URL;
  if (override) return override;

  if (typeof window === "undefined") return "ws://localhost:8000/ws";

  const { protocol, hostname, host } = window.location;
  const scheme = protocol === "https:" ? "wss" : "ws";

  if (isLocalHost(hostname)) return `${scheme}://${hostname}:8000/ws`;

  if (host === new URL(RENDER_BACKEND).host) return `${scheme}://${host}/ws`;

  return `${RENDER_BACKEND.replace(/^http/, "ws")}/ws`;
}
