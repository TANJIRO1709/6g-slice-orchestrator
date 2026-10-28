import re, shutil, ast
from pathlib import Path

MAIN = Path("api/main.py")
if not MAIN.exists():
    raise SystemExit("api/main.py not found. Run from backend/")

src = MAIN.read_text()
if "agents_enabled" in src:
    raise SystemExit("Already patched.")

shutil.copy(MAIN, "api/main.py.bak")
print("Backed up to api/main.py.bak")

src = src.replace('strategy = "equal"',
    'strategy = "equal"\npipeline = None\nagents_enabled = False\nlatest_trace = None\ntrace_log = []', 1)

LOAD = '''
def load_pipeline():
    global pipeline
    import os, traceback
    backend_dir = Path(__file__).resolve().parent.parent
    prev = os.getcwd()
    try:
        os.chdir(backend_dir)
        from agents.pipeline import AgenticPipeline
        pipeline = AgenticPipeline(use_llm=True, use_rag=True, diagnose_every=5)
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


'''
src = src.replace("def get_action(", LOAD.lstrip("\n") + "def get_action(", 1)
src = src.replace("    load_drl_model()", "    load_drl_model()\n    load_pipeline()", 1)

NEW_STEP = '''def run_one_step() -> dict:
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
'''
pat = re.compile(r"def run_one_step\(\).*?\n    return state\n", re.DOTALL)
if not pat.search(src):
    raise SystemExit("Could not find run_one_step()")
src = pat.sub(NEW_STEP, src, count=1)

EP = '''
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


'''
if '@app.websocket("/ws")' in src:
    src = src.replace('@app.websocket("/ws")', EP.lstrip("\n") + '@app.websocket("/ws")', 1)
else:
    src += EP

rp = re.compile(r'(@app\.post\("/api/reset"\)\s*\nasync def reset\(\):.*?\n)(\s*)sim\.reset\(\)\n', re.DOTALL)
if rp.search(src):
    src = rp.sub(lambda m: m.group(1)+m.group(2)+"sim.reset()\n"+m.group(2)+"if pipeline:\n"+m.group(2)+"    pipeline.reset()\n"+m.group(2)+"trace_log.clear()\n", src, count=1)

MAIN.write_text(src)
try:
    ast.parse(src)
except SyntaxError as e:
    shutil.copy("api/main.py.bak", MAIN)
    raise SystemExit(f"Invalid syntax ({e}). Restored backup.")

print(f"Patched. agents_enabled count: {src.count('agents_enabled')}")
