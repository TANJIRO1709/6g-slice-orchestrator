/**
 * Resolves where the backend lives.
 *
 * Local development: Next runs on :3000, FastAPI on :8000, so
 * requests are sent explicitly to :8000.
 *
 * Deployed: FastAPI serves the compiled dashboard itself, so the
 * backend is the same origin. apiBase() returns "" and fetches
 * use relative paths like "/api/start".
 *
 * Either can be overridden with NEXT_PUBLIC_API_URL /
 * NEXT_PUBLIC_WS_URL when the two are deployed separately.
 */

function isLocalHost(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1";
}

export function apiBase(): string {
  const override = process.env.NEXT_PUBLIC_API_URL;
  if (override) return override.replace(/\/$/, "");

  if (typeof window === "undefined") return "http://localhost:8000";

  const { protocol, hostname } = window.location;
  if (isLocalHost(hostname)) return `${protocol}//${hostname}:8000`;

  return ""; // same origin — relative paths
}

export function wsUrl(): string {
  const override = process.env.NEXT_PUBLIC_WS_URL;
  if (override) return override;

  if (typeof window === "undefined") return "ws://localhost:8000/ws";

  const { protocol, hostname, host } = window.location;
  const scheme = protocol === "https:" ? "wss" : "ws";

  if (isLocalHost(hostname)) return `${scheme}://${hostname}:8000/ws`;

  return `${scheme}://${host}/ws`;
}
