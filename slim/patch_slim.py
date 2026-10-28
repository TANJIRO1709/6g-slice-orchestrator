#!/usr/bin/env python3
"""
Switch the API to NumPy policy inference so torch is not needed to serve.

Run from the PROJECT ROOT, after patch_deploy.py:

    python3 patch_slim.py

Rewrites load_drl_model() in backend/api/main.py to prefer
models/policy_weights.npz, falling back to the stable-baselines3
checkpoint when torch happens to be installed. Local development is
unaffected either way.
"""

import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
MAIN = ROOT / "backend" / "api" / "main.py"

if not MAIN.exists():
    print(f"  ERROR: {MAIN} not found. Run this from the project root.")
    sys.exit(1)

src = MAIN.read_text()

if "NumpyPolicy" in src:
    print("  Already patched — nothing to do.")
    sys.exit(0)

NEW_LOADER = '''def load_drl_model():
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
                "⚠️  Found a checkpoint but stable-baselines3 is not installed.\\n"
                "    Export NumPy weights instead:  python export_policy.py"
            )
            return
        except Exception as e:
            print(f"⚠️  Could not load {name}: {e}")

    print("ℹ️  No trained policy found — the AI (PPO) strategy will be inert.")
    print("   Train:  python -m rl.train --timesteps 200000 --eval")
    print("   Export: python export_policy.py")


'''

# ── Replace load_drl_model() ─────────────────────────────
pat = re.compile(r"^def load_drl_model\(\):.*?(?=^def |^@|^class )", re.MULTILINE | re.DOTALL)
if not pat.search(src):
    print("  ERROR: could not locate load_drl_model() in api/main.py")
    sys.exit(1)
# A lambda replacement, not a plain string: re.sub would otherwise
# interpret backslash escapes inside NEW_LOADER and corrupt it.
src = pat.sub(lambda _m: NEW_LOADER, src, count=1)

# ── get_action: accept both return shapes ────────────────
old_call = """        action, _ = drl_model.predict(obs, deterministic=True)
        return action"""
new_call = """        result = drl_model.predict(obs, deterministic=True)
        # stable-baselines3 returns (action, state); NumpyPolicy returns
        # the action on its own.
        return result[0] if isinstance(result, tuple) else result"""

if old_call in src:
    src = src.replace(old_call, new_call, 1)
else:
    print("  WARNING: get_action() did not match the expected shape.")
    print("           Check that the drl branch handles a bare ndarray.")

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"  ERROR: patched main.py is not valid Python ({e}). Nothing written.")
    sys.exit(1)

shutil.copy(MAIN, MAIN.with_suffix(".py.slim.bak"))
MAIN.write_text(src)

print("=" * 58)
print("  Slim serving patch applied")
print("=" * 58)
print("    + backend/api/main.py — NumPy policy preferred")
print("    + backup at backend/api/main.py.slim.bak")
print()
print("  Next:")
print("    cd backend && ./venv/bin/python export_policy.py")
print("    cd .. && docker build -f Dockerfile.slim -t 6g-orchestrator .")
print()
