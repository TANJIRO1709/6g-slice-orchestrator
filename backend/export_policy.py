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

If torch / stable-baselines3 are not installed, the weights are read
straight out of the checkpoint zip with NumPy instead (the torch-vs-
NumPy verification step is skipped in that case).
"""

import io
import json
import pickle
import sys
import zipfile
from collections import OrderedDict
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

# torch storage class name -> NumPy dtype, for the torch-free reader
_STORAGE_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "HalfStorage": np.float16, "LongStorage": np.int64,
    "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor(storage, offset, size, stride, *_):
    itemsize = storage.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size),
        strides=tuple(s * itemsize for s in stride),
    ).copy()


def read_without_torch(model_path: Path):
    """
    Read the policy state_dict and spaces from an SB3 checkpoint zip
    using only NumPy. Returns (state, action_low, action_high, obs_dim).
    """
    with zipfile.ZipFile(model_path) as outer:
        meta = json.loads(outer.read("data"))
        pth = zipfile.ZipFile(io.BytesIO(outer.read("policy.pth")))

    root = pth.namelist()[0].split("/")[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in _STORAGE_DTYPES:
                return _StorageType(name)
            if name.startswith("_rebuild_tensor"):
                return _rebuild_tensor
            if (module, name) == ("collections", "OrderedDict"):
                return OrderedDict
            raise pickle.UnpicklingError(f"unexpected global {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, _location, _numel = pid
            raw = pth.read(f"{root}/data/{key}")
            return np.frombuffer(raw, dtype=_STORAGE_DTYPES[storage_type.name])

    state = Unpickler(io.BytesIO(pth.read(f"{root}/data.pkl"))).load()

    def parse(text):
        return np.array(text.strip("[]").split(), dtype=np.float32)

    action = meta["action_space"]
    obs_dim = meta["observation_space"]["_shape"][0]
    return state, parse(action["low"]), parse(action["high"]), obs_dim


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
        PPO = None

    print(f"Loading {model_path}")
    if PPO is None:
        print("stable-baselines3 not installed — reading the checkpoint with NumPy")
        model = None
        state, low, high, obs_dim = read_without_torch(model_path)
    else:
        model = PPO.load(str(model_path), device="cpu")
        state = model.policy.state_dict()
        low = np.asarray(model.action_space.low, dtype=np.float32)
        high = np.asarray(model.action_space.high, dtype=np.float32)
        obs_dim = model.observation_space.shape[0]

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

    if model is None:
        weights = {k: np.asarray(state[k], dtype=np.float32) for k in POLICY_KEYS}
    else:
        weights = {k: state[k].detach().cpu().numpy().astype(np.float32) for k in POLICY_KEYS}

    weights["_action_low"] = low
    weights["_action_high"] = high
    weights["_obs_dim"] = np.array([obs_dim], dtype=np.int32)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **weights)

    size_kb = OUT.stat().st_size / 1024
    print(f"\nWrote {OUT}  ({size_kb:.0f} KB)")
    print(f"  observation dim : {int(weights['_obs_dim'][0])}")
    print(f"  action bounds   : [{low.min()}, {high.max()}]")

    sys.path.insert(0, str(BACKEND))
    from rl.policy_numpy import NumpyPolicy

    policy = NumpyPolicy(OUT)
    rng = np.random.RandomState(0)

    if model is None:
        action = policy.predict(rng.randn(obs_dim).astype(np.float32))
        print(f"\nNumPy policy loads and predicts: {action}")
        print("(torch not installed, so the torch comparison was skipped)")
        print("\nNext: commit models/policy_weights.npz and deploy.")
        return 0

    # ── Verify the NumPy path reproduces torch ───────────
    print("\nVerifying NumPy inference against torch...")

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
