#!/usr/bin/env python3
"""
Export the trained PPO policy to a NumPy archive.

Run this ONCE on your own machine, where torch and stable-baselines3
are installed:

    cd backend
    ./venv/bin/python export_policy.py

It writes backend/models/policy_weights.npz — a few hundred kilobytes
that the deployed container loads with NumPy alone. torch and
stable-baselines3 then drop out of the deployment image entirely,
taking it from roughly 1.2 GB to under 400 MB and bringing 512 MB
free tiers into range.

Training still uses torch as before; only serving changes.
"""

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent
CANDIDATES = [
    BACKEND / "models" / "best" / "best_model.zip",
    BACKEND / "models" / "ppo_slice_allocator_final.zip",
]
OUT = BACKEND / "models" / "policy_weights.npz"

# Keys that make up the deterministic action path. The value network
# is only needed for training, so it is left out.
POLICY_KEYS = [
    "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
    "mlp_extractor.policy_net.4.weight", "mlp_extractor.policy_net.4.bias",
    "action_net.weight", "action_net.bias",
]


def main() -> int:
    model_path = next((p for p in CANDIDATES if p.exists()), None)
    if model_path is None:
        print("ERROR: no trained model found. Looked for:")
        for p in CANDIDATES:
            print(f"  {p}")
        print("\nTrain one first:")
        print("  ./venv/bin/python -m rl.train --timesteps 200000 --eval")
        return 1

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("ERROR: stable-baselines3 is not installed in this environment.")
        print("Run this with the venv python that you trained with.")
        return 1

    print(f"Loading {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    state = model.policy.state_dict()

    missing = [k for k in POLICY_KEYS if k not in state]
    if missing:
        print("ERROR: the saved policy does not have the expected layers.")
        print("Missing:", missing)
        print("\nFound instead:")
        for k in state:
            print("  ", k)
        print(
            "\nThis exporter assumes net_arch pi=[128, 128, 64], which is what\n"
            "rl/train.py configures. If you changed the architecture, update\n"
            "POLICY_KEYS and the layer count in rl/policy_numpy.py to match."
        )
        return 1

    weights = {k: state[k].detach().cpu().numpy().astype(np.float32) for k in POLICY_KEYS}

    low = np.asarray(model.action_space.low, dtype=np.float32)
    high = np.asarray(model.action_space.high, dtype=np.float32)
    weights["_action_low"] = low
    weights["_action_high"] = high
    weights["_obs_dim"] = np.array([model.observation_space.shape[0]], dtype=np.int32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **weights)

    size_kb = OUT.stat().st_size / 1024
    print(f"\nWrote {OUT}  ({size_kb:.0f} KB)")
    print(f"  observation dim : {int(weights['_obs_dim'][0])}")
    print(f"  action bounds   : [{low.min()}, {high.max()}]")

    # ── Verify the NumPy path reproduces torch ───────────
    print("\nVerifying NumPy inference against torch...")
    sys.path.insert(0, str(BACKEND))
    from rl.policy_numpy import NumpyPolicy

    policy = NumpyPolicy(OUT)
    rng = np.random.RandomState(0)
    obs_dim = int(weights["_obs_dim"][0])

    worst = 0.0
    for _ in range(200):
        obs = rng.randn(obs_dim).astype(np.float32) * 2.0
        torch_action, _ = model.predict(obs, deterministic=True)
        numpy_action = policy.predict(obs)
        worst = max(worst, float(np.abs(torch_action - numpy_action).max()))

    if worst < 1e-5:
        print(f"  Match confirmed — largest difference {worst:.2e} over 200 samples")
        print("\nNext: commit models/policy_weights.npz and deploy.")
        return 0

    print(f"  MISMATCH — largest difference {worst:.2e}")
    print("  Do not deploy the slim image; the NumPy forward pass is wrong.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
