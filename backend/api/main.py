"""
FastAPI backend for the 6G Network Slice Orchestrator.

Provides:
- REST endpoints for simulation control and metrics
- WebSocket for real-time dashboard updates
- Integration with DRL agent for live allocation decisions
"""

import asyncio
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from simulator.network import NetworkSimulator
from api.schemas import SimulationConfig, SimulationState, ActionRequest

# ── Global State ─────────────────────────────────────────────────

sim: Optional[NetworkSimulator] = None
drl_model = None
is_running = False
strategy = "equal"
pipeline = None
agents_enabled = False
latest_trace = None
trace_log = []
sim_speed = 1.0
connected_clients: Set[WebSocket] = set()
agent_log: List[dict] = []


def load_drl_model():
    """
    Load the trained PPO policy.

    Prefers models/policy_weights.npz, which is served with NumPy alone
    and keeps torch out of the deployment image. Falls back to the
    stable-baselines3 checkpoint when torch is available, so local
    development works whether or not the weights have been exported.
    """
    global drl_model

    backend_dir = Path(__file__).resolve().parent.parent

    # ── Preferred: NumPy weights, no torch required ──────
    npz = backend_dir / "models" / "policy_weights.npz"
    if npz.exists():
        try:
            from rl.policy_numpy import NumpyPolicy

            drl_model = NumpyPolicy(npz)
            print(f"✅ Loaded DRL policy (NumPy) from {npz.name}")
            return
        except Exception as e:
            print(f"⚠️  NumPy policy failed to load: {e}")

    # ── Fallback: full stable-baselines3 checkpoint ──────
    for name in ("models/best/best_model.zip", "models/ppo_slice_allocator_final.zip"):
        path = backend_dir / name
        if not path.exists():
            continue
        try:
            from stable_baselines3 import PPO

            drl_model = PPO.load(str(path), device="cpu")
            print(f"✅ Loaded DRL model from {name}")
            return
        except ImportError:
            print(
                "⚠️  Found a checkpoint but stable-baselines3 is not installed.\n"
                "    Export NumPy weights instead:  python export_policy.py"
            )
            return
        except Exception as e:
            print(f"⚠️  Could not load {name}: {e}")

    print("ℹ️  No trained policy found — the AI (PPO) strategy will be inert.")
    print("   Train:  python -m rl.train --timesteps 200000 --eval")
    print("   Export: python export_policy.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global sim
    sim = NetworkSimulator(total_bandwidth_mhz=100.0, seed=42)
    sim.reset()
    load_drl_model()
    load_pipeline()
    print("\n🚀 6G Slice Orchestrator API ready at http://localhost:8000")
    print("📊 Open dashboard at http://localhost:3000\n")
    yield
    print("Shutting down...")


# ── App Setup ────────────────────────────────────────────────────

app = FastAPI(
    title="6G Network Slice Orchestrator",
    description="LLM-Powered Agentic AI for 6G Slice Management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Strategy Functions ───────────────────────────────────────────

def load_pipeline():
    global pipeline
    import os, traceback
    backend_dir = Path(__file__).resolve().parent.parent
    prev = os.getcwd()
    try:
        os.chdir(backend_dir)
        from agents.pipeline import AgenticPipeline
        pipeline = AgenticPipeline(
            use_llm=True,
            use_rag=True,
            diagnose_every=int(os.getenv("DIAGNOSE_EVERY", "5")),
        )
        s = pipeline.get_stats()
        print(f"Agentic pipeline ready (mode: {s['mode']}, RAG: {s['rag_enabled']})")
    except Exception as e:
        print("="*60)
        print(f"Agentic pipeline FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("="*60)
        pipeline = None
    finally:
        os.chdir(prev)


def get_action(obs: np.ndarray) -> np.ndarray:
    """Get allocation action based on selected strategy."""
    global strategy
    if strategy == "drl" and drl_model is not None:
        result = drl_model.predict(obs, deterministic=True)
        # stable-baselines3 returns (action, state); NumpyPolicy returns
        # the action on its own.
        return result[0] if isinstance(result, tuple) else result
    elif strategy == "urllc_heavy":
        return np.array([0.0, 1.5, -0.5])
    elif strategy == "embb_heavy":
        return np.array([1.5, 0.0, -0.5])
    else:  # "equal" or fallback
        return np.array([0.0, 0.0, 0.0])


def run_one_step() -> dict:
    global latest_trace
    metrics = sim.get_metrics_snapshot()
    obs = sim._get_observation()
    raw_action = get_action(obs)
    trace = None
    action = raw_action
    if agents_enabled and pipeline is not None:
        try:
            action, trace = pipeline.process(metrics, raw_action, sim.history)
            latest_trace = trace.to_dict()
            if trace.diagnosis:
                trace_log.append(latest_trace)
                if len(trace_log) > 100:
                    del trace_log[:50]
        except Exception as e:
            print(f"Pipeline error: {e}")
            action = raw_action
    obs, reward, done, info = sim.step(action)
    state = sim.get_metrics_snapshot()
    state["reward"] = round(reward, 3)
    state["strategy"] = strategy
    state["is_running"] = is_running
    state["agents_enabled"] = agents_enabled
    if trace is not None:
        state["agent"] = {
            "alerts": trace.alerts,
            "alert_count": len(trace.alerts),
            "diagnosis": trace.diagnosis,
            "decision": trace.decision,
            "latency_ms": trace.latency_ms,
            "used_llm": trace.used_llm,
            "llm_calls": trace.llm_calls,
        }
        state["agent_stats"] = pipeline.get_stats()
    else:
        state["agent"] = None
    return state


# ── REST Endpoints ───────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "6G Slice Orchestrator"}


@app.get("/api/state")
async def get_state():
    """Get current simulation state."""
    state = sim.get_metrics_snapshot()
    state["is_running"] = is_running
    state["strategy"] = strategy
    return state


@app.get("/api/history")
async def get_history(last_n: int = 200):
    return sim.history[-last_n:]


@app.get("/api/agent-log")
async def get_agent_log(last_n: int = 50):
    return agent_log[-last_n:]


@app.post("/api/start")
async def start_simulation():
    global is_running
    is_running = True
    return {"status": "running"}


@app.post("/api/pause")
async def pause_simulation():
    global is_running
    is_running = False
    return {"status": "paused"}


@app.post("/api/reset")
async def reset_simulation():
    global is_running, agent_log
    is_running = False
    sim.reset()
    agent_log = []
    return {"status": "reset"}


@app.post("/api/step")
async def manual_step():
    state = run_one_step()
    await broadcast(state)
    return state


@app.post("/api/strategy/{name}")
async def set_strategy(name: str):
    global strategy
    if name in ("equal", "drl", "embb_heavy", "urllc_heavy"):
        strategy = name
        return {"status": "ok", "strategy": strategy}
    raise HTTPException(400, "Invalid strategy")


@app.post("/api/speed/{value}")
async def set_speed(value: float):
    global sim_speed
    sim_speed = max(0.1, min(value, 20.0))
    return {"status": "ok", "speed": sim_speed}


@app.get("/api/strategies")
async def list_strategies():
    return [
        {"id": "equal", "name": "Equal Static", "available": True},
        {"id": "drl", "name": "DRL (PPO)", "available": drl_model is not None},
        {"id": "embb_heavy", "name": "eMBB Priority", "available": True},
        {"id": "urllc_heavy", "name": "URLLC Priority", "available": True},
    ]


# ── WebSocket ────────────────────────────────────────────────────

async def broadcast(data: dict):
    """Send data to all connected WebSocket clients."""
    if not connected_clients:
        return
    dead = set()
    for ws in connected_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    connected_clients.difference_update(dead)


async def ws_receiver(websocket: WebSocket):
    """
    Task that listens for commands from the frontend.
    Runs in its own coroutine so it never blocks the sender.
    """
    global is_running, strategy, sim_speed

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            cmd = msg.get("command", "")
            print(f"  ← Received command: {cmd}")

            if cmd == "start":
                is_running = True
            elif cmd == "pause":
                is_running = False
            elif cmd == "reset":
                is_running = False
                sim.reset()
            elif cmd == "step":
                state = run_one_step()
                await broadcast(state)
            elif cmd == "set_strategy":
                strategy = msg.get("strategy", "equal")
                print(f"    Strategy changed to: {strategy}")
            elif cmd == "set_speed":
                sim_speed = float(msg.get("speed", 1.0))
                print(f"    Speed changed to: {sim_speed}x")

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


async def ws_sender(websocket: WebSocket):
    """
    Task that continuously sends simulation state to the frontend.
    Runs simulation steps when is_running is True.
    """
    try:
        while True:
            if is_running:
                # Run a simulation step
                state = run_one_step()
                await broadcast(state)

                # Control speed: higher sim_speed = shorter sleep
                delay = max(0.03, 0.3 / sim_speed)
                await asyncio.sleep(delay)
            else:
                # When paused, send current state periodically as heartbeat
                state = sim.get_metrics_snapshot()
                state["is_running"] = False
                state["strategy"] = strategy
                state["reward"] = 0
                await websocket.send_json(state)
                await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass


@app.post("/api/agents/{enabled}")
async def toggle_agents(enabled: str):
    global agents_enabled
    agents_enabled = enabled.lower() in ("true", "on", "1", "enable")
    print(f"Agents: {'ON' if agents_enabled else 'OFF'}")
    return {"agents_enabled": agents_enabled}


@app.get("/api/agents/status")
async def agents_status():
    if pipeline is None:
        return {"available": False, "reason": "pipeline failed to initialise"}
    return {"available": True, "enabled": agents_enabled, **pipeline.get_stats()}


@app.get("/api/agents/traces")
async def agent_traces(last_n: int = 20):
    return trace_log[-last_n:]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint — spawns two tasks:
    1. Receiver: listens for commands (start, pause, strategy change)
    2. Sender: pushes simulation state to the frontend
    
    This avoids the broken wait_for/timeout pattern.
    """
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔌 Dashboard connected. Clients: {len(connected_clients)}")

    # Send initial state immediately
    state = sim.get_metrics_snapshot()
    state["is_running"] = is_running
    state["strategy"] = strategy
    state["reward"] = 0
    await websocket.send_json(state)

    # Run receiver and sender as parallel tasks
    receiver_task = asyncio.create_task(ws_receiver(websocket))
    sender_task = asyncio.create_task(ws_sender(websocket))

    # Wait for either to finish (meaning disconnect)
    done, pending = await asyncio.wait(
        [receiver_task, sender_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel the other task
    for task in pending:
        task.cancel()

    connected_clients.discard(websocket)
    print(f"🔌 Dashboard disconnected. Clients: {len(connected_clients)}")


# ── Run ──────────────────────────────────────────────────────────

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)