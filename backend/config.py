"""
Global configuration for the 6G Network Slice Orchestrator.
All simulation, training, and system parameters in one place.
"""

import numpy as np

# ── Network Configuration ────────────────────────────────────────
TOTAL_BANDWIDTH_MHZ = 100.0        # Total available bandwidth in MHz
TOTAL_POWER_DBM = 46.0             # Base station transmit power in dBm
NUM_SUBCARRIERS = 100              # Resource blocks to allocate
CARRIER_FREQ_GHZ = 3.5             # Carrier frequency
SIMULATION_STEP_MS = 1.0           # Duration of each simulation step

# ── Slice Definitions ────────────────────────────────────────────
SLICE_CONFIGS = {
    "eMBB": {
        "id": 0,
        "num_users": 20,
        "min_throughput_mbps": 100.0,
        "max_latency_ms": 10.0,
        "reliability": 0.999,
        "priority": 1,
        "color": "#3b82f6",  # blue
    },
    "URLLC": {
        "id": 1,
        "num_users": 10,
        "min_throughput_mbps": 1.0,
        "max_latency_ms": 1.0,
        "reliability": 0.99999,
        "priority": 3,  # highest priority
        "color": "#ef4444",  # red
    },
    "mMTC": {
        "id": 2,
        "num_users": 50,
        "min_throughput_mbps": 0.1,
        "max_latency_ms": 100.0,
        "reliability": 0.99,
        "priority": 2,
        "color": "#22c55e",  # green
    },
}

NUM_SLICES = len(SLICE_CONFIGS)

# ── Channel Model ────────────────────────────────────────────────
NOISE_POWER_DBM = -174.0 + 10 * np.log10(TOTAL_BANDWIDTH_MHZ * 1e6)
PATH_LOSS_EXPONENT = 3.5
CELL_RADIUS_M = 500.0
RAYLEIGH_SCALE = 1.0
CHANNEL_UPDATE_INTERVAL = 1        # Update channel every N steps

# ── DRL Training ─────────────────────────────────────────────────
DRL_CONFIG = {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "total_timesteps": 500_000,
    "eval_freq": 10_000,
    "model_save_path": "models/",
}

# ── Reward Weights ───────────────────────────────────────────────
REWARD_WEIGHTS = {
    "embb_violation": -2.0,
    "urllc_violation": -10.0,    # highest penalty — most critical
    "mmtc_violation": -1.0,
    "utilization_bonus": 1.0,
    "stability_bonus": 0.5,      # reward for not changing allocation too much
}

# ── API / Dashboard ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
WS_BROADCAST_INTERVAL_MS = 500
MAX_HISTORY_LENGTH = 1000          # Max data points to keep in memory
