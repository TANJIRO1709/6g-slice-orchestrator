#!/usr/bin/env python3
"""
Prepare the 6G Slice Orchestrator for deployment.

Run from the PROJECT ROOT (the folder containing backend/ and frontend/):

    python3 patch_deploy.py

Changes made — all idempotent, with .bak backups:

  backend/api/main.py
      - serves the compiled dashboard from ./static (mounted last
        so it never shadows /api or /ws)
      - reads DIAGNOSE_EVERY from the environment so LLM call rate
        can be throttled in production

  frontend/next.config.js
      - static export so the dashboard compiles to plain files

  frontend/src/lib/config.ts            (new)
      - resolves API and WebSocket URLs for local vs deployed

  frontend/src/hooks/useWebSocket.ts
  frontend/src/app/page.tsx
      - use the resolver instead of hardcoded localhost:8000

  .gitignore
      - stop excluding the trained PPO model, which must ship
"""

import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"

changes: list[str] = []
skipped: list[str] = []


def fail(msg: str) -> None:
    print(f"\n  ERROR: {msg}")
    sys.exit(1)


if not BACKEND.is_dir() or not FRONTEND.is_dir():
    fail(
        "Run this from the project root — the folder that contains both\n"
        f"         backend/ and frontend/. Current directory: {ROOT}"
    )


def backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        shutil.copy(p, bak)


# ══════════════════════════════════════════════════════════
# 1. backend/api/main.py — static mount + DIAGNOSE_EVERY
# ══════════════════════════════════════════════════════════

main_py = BACKEND / "api" / "main.py"
if not main_py.exists():
    fail(f"{main_py} not found")

src = main_py.read_text()
original = src

if "StaticFiles" not in src:
    STATIC_BLOCK = '''
# ── Serve the compiled dashboard ─────────────────────────
# Mounted last so it never shadows /api/* or /ws. In local
# development ./static does not exist and this is skipped —
# the Next dev server handles the UI on port 3000 instead.

_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="dashboard")
    print(f"🖥️  Serving dashboard from {_static_dir}")
else:
    print("ℹ️  No ./static directory — run the Next dev server for the UI")


'''
    anchor = 'if __name__ == "__main__":'
    if anchor not in src:
        fail("could not find the __main__ block in api/main.py")
    src = src.replace(anchor, STATIC_BLOCK.lstrip("\n") + anchor, 1)
    changes.append("backend/api/main.py — mounts ./static")
else:
    skipped.append("backend/api/main.py static mount (already present)")

# DIAGNOSE_EVERY from the environment
if "DIAGNOSE_EVERY" not in src:
    pat = re.compile(
        r"AgenticPipeline\(\s*use_llm=True,\s*use_rag=True,\s*diagnose_every=\d+\s*\)"
    )
    if pat.search(src):
        src = pat.sub(
            "AgenticPipeline(\n"
            "            use_llm=True,\n"
            "            use_rag=True,\n"
            '            diagnose_every=int(os.getenv("DIAGNOSE_EVERY", "5")),\n'
            "        )",
            src,
            count=1,
        )
        # `import os` is already inside load_pipeline() in the patched
        # build; add a module-level import when it is missing.
        if not re.search(r"^import os$", src, re.MULTILINE):
            src = src.replace("import asyncio", "import asyncio\nimport os", 1)
        changes.append("backend/api/main.py — DIAGNOSE_EVERY from env")
    else:
        skipped.append("DIAGNOSE_EVERY (AgenticPipeline call not matched)")
else:
    skipped.append("DIAGNOSE_EVERY (already present)")

if src != original:
    backup(main_py)
    try:
        ast.parse(src)
    except SyntaxError as e:
        fail(f"patched main.py is not valid Python ({e}); nothing written")
    main_py.write_text(src)


# ══════════════════════════════════════════════════════════
# 2. frontend/next.config.js — static export
# ══════════════════════════════════════════════════════════

next_cfg = FRONTEND / "next.config.js"
NEXT_CONFIG = """/** @type {import('next').NextConfig} */
const nextConfig = {
  // Compile to plain static files so FastAPI can serve them.
  output: 'export',
  images: { unoptimized: true },
  trailingSlash: true,
};

module.exports = nextConfig;
"""
if not next_cfg.exists() or "output: 'export'" not in next_cfg.read_text():
    if next_cfg.exists():
        backup(next_cfg)
    next_cfg.write_text(NEXT_CONFIG)
    changes.append("frontend/next.config.js — static export")
else:
    skipped.append("next.config.js (already exporting)")


# ══════════════════════════════════════════════════════════
# 3. frontend/src/lib/config.ts — URL resolver
# ══════════════════════════════════════════════════════════

lib_dir = FRONTEND / "src" / "lib"
lib_dir.mkdir(parents=True, exist_ok=True)
cfg_ts = lib_dir / "config.ts"

CONFIG_TS = '''/**
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
  if (override) return override.replace(/\\/$/, "");

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
'''

if not cfg_ts.exists():
    cfg_ts.write_text(CONFIG_TS)
    changes.append("frontend/src/lib/config.ts — created")
else:
    skipped.append("config.ts (already exists)")


# ══════════════════════════════════════════════════════════
# 4. useWebSocket.ts — use the resolver
# ══════════════════════════════════════════════════════════

ws_ts = FRONTEND / "src" / "hooks" / "useWebSocket.ts"
if not ws_ts.exists():
    fail(f"{ws_ts} not found")

s = ws_ts.read_text()
if "wsUrl" not in s:
    backup(ws_ts)

    s = re.sub(
        r'^const WS_URL = "ws://localhost:8000/ws";\s*$',
        "",
        s,
        count=1,
        flags=re.MULTILINE,
    )
    s = s.replace(
        'import { useEffect, useRef, useState, useCallback } from "react";',
        'import { useEffect, useRef, useState, useCallback } from "react";\n'
        'import { wsUrl } from "@/lib/config";',
        1,
    )
    s = s.replace("new WebSocket(WS_URL)", "new WebSocket(wsUrl())")

    if "wsUrl()" not in s:
        fail("could not rewrite the WebSocket URL in useWebSocket.ts")

    ws_ts.write_text(s)
    changes.append("frontend/src/hooks/useWebSocket.ts — dynamic WS URL")
else:
    skipped.append("useWebSocket.ts (already dynamic)")


# ══════════════════════════════════════════════════════════
# 5. page.tsx — use the resolver
# ══════════════════════════════════════════════════════════

page_tsx = FRONTEND / "src" / "app" / "page.tsx"
if not page_tsx.exists():
    fail(f"{page_tsx} not found")

s = page_tsx.read_text()
if "apiBase" not in s:
    backup(page_tsx)

    s = s.replace(
        'import AgentPanel from "@/components/AgentPanel";',
        'import AgentPanel from "@/components/AgentPanel";\n'
        'import { apiBase } from "@/lib/config";',
        1,
    )
    s = re.sub(
        r'\n\s*const API = "http://localhost:8000";\s*\n',
        "\n",
        s,
        count=1,
    )
    s = s.replace("fetch(`${API}${path}`", "fetch(`${apiBase()}${path}`")

    if "apiBase()" not in s:
        fail("could not rewrite the API base in page.tsx")

    page_tsx.write_text(s)
    changes.append("frontend/src/app/page.tsx — dynamic API base")
else:
    skipped.append("page.tsx (already dynamic)")


# ══════════════════════════════════════════════════════════
# 6. .gitignore — let the trained model ship
# ══════════════════════════════════════════════════════════

for gi in (ROOT / ".gitignore", BACKEND / ".gitignore"):
    if not gi.exists():
        continue
    text = gi.read_text()
    if "!backend/models/best/best_model.zip" in text:
        continue
    out, touched = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in ("models/", "models", "*.zip"):
            out.append(f"# {line}        # unignored for deployment")
            touched = True
        else:
            out.append(line)
    if touched:
        out += [
            "",
            "# Training output stays out; the trained policy must ship.",
            "backend/models/",
            "!backend/models/best/",
            "!backend/models/best/best_model.zip",
            "!backend/models/ppo_slice_allocator_final.zip",
        ]
        backup(gi)
        gi.write_text("\n".join(out) + "\n")
        changes.append(f"{gi.relative_to(ROOT)} — trained model no longer ignored")


# ══════════════════════════════════════════════════════════

print("\n" + "=" * 58)
print("  Deployment prep")
print("=" * 58)
if changes:
    print("\n  Applied:")
    for c in changes:
        print(f"    + {c}")
if skipped:
    print("\n  Already done:")
    for c in skipped:
        print(f"    . {c}")

print("\n  Next:")
print("    1. Confirm the trained model is tracked by git:")
print("       git add -f backend/models/best/best_model.zip")
print("    2. Build locally to check:")
print("       docker build -t 6g-orchestrator .")
print("    3. Run it:")
print("       docker run -p 7860:7860 -e GEMINI_API_KEY=<key> 6g-orchestrator")
print()
