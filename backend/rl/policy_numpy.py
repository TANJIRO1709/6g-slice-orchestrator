"""
NumPy-only inference for the trained PPO policy.

Reproduces exactly what Stable-Baselines3 does for
`model.predict(obs, deterministic=True)` on a Box action space:

    x = obs
    x = tanh(W0 x + b0)
    x = tanh(W2 x + b2)
    x = tanh(W4 x + b4)
    a = W_action x + b_action
    a = clip(a, action_low, action_high)

For a deterministic Box action SB3 returns the distribution mode,
which for DiagGaussianDistribution is the mean — the raw action_net
output, with no sampling and no log_std involvement.

Weights come from models/policy_weights.npz, produced by
export_policy.py. This module imports nothing but NumPy, which is
what lets torch and stable-baselines3 stay out of the deployed image.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "policy_weights.npz"

# (weight key, bias key) for each Tanh-activated hidden layer
_HIDDEN_LAYERS = [
    ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias"),
    ("mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias"),
    ("mlp_extractor.policy_net.4.weight", "mlp_extractor.policy_net.4.bias"),
]


class NumpyPolicy:
    """Drop-in replacement for a loaded PPO model's deterministic predict()."""

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Generate it on a machine that has torch:\n"
                f"    ./venv/bin/python export_policy.py"
            )

        data = np.load(path)

        self.layers = [
            (data[w].astype(np.float32), data[b].astype(np.float32))
            for w, b in _HIDDEN_LAYERS
        ]
        self.out_w = data["action_net.weight"].astype(np.float32)
        self.out_b = data["action_net.bias"].astype(np.float32)

        self.action_low = data["_action_low"].astype(np.float32)
        self.action_high = data["_action_high"].astype(np.float32)
        self.obs_dim = int(data["_obs_dim"][0])

        self.path = path

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Return the deterministic action for a single observation.

        `deterministic` is accepted so call sites read the same as the
        SB3 version; this policy is always deterministic.
        """
        x = np.asarray(obs, dtype=np.float32).reshape(-1)

        if x.shape[0] != self.obs_dim:
            raise ValueError(
                f"expected an observation of length {self.obs_dim}, got {x.shape[0]}"
            )

        for w, b in self.layers:
            x = np.tanh(x @ w.T + b)

        action = x @ self.out_w.T + self.out_b
        return np.clip(action, self.action_low, self.action_high)

    def __repr__(self) -> str:
        return f"NumpyPolicy(obs_dim={self.obs_dim}, path={self.path.name})"


def load_policy(weights_path: Optional[Union[str, Path]] = None) -> Optional[NumpyPolicy]:
    """Load the policy, returning None rather than raising if unavailable."""
    try:
        return NumpyPolicy(weights_path)
    except FileNotFoundError:
        return None
