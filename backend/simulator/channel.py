"""
Wireless channel model for the 6G network simulator.
Models path loss + Rayleigh fading for each user-to-BS link.
"""

import numpy as np
from typing import Dict, List


class ChannelModel:
    """
    Models the wireless channel between users and the base station.
    
    Each user has:
    - A fixed distance-based path loss
    - A time-varying Rayleigh fading component (small-scale)
    - Resulting SNR that changes every update interval
    """

    def __init__(
        self,
        num_users_per_slice: Dict[str, int],
        cell_radius_m: float = 500.0,
        carrier_freq_ghz: float = 3.5,
        path_loss_exponent: float = 3.5,
        noise_power_dbm: float = -100.0,
        bs_power_dbm: float = 46.0,
        seed: int = None,
    ):
        self.cell_radius = cell_radius_m
        self.carrier_freq = carrier_freq_ghz
        self.path_loss_exp = path_loss_exponent
        self.noise_power_dbm = noise_power_dbm
        self.bs_power_dbm = bs_power_dbm
        self.rng = np.random.RandomState(seed)

        # Total users across all slices
        self.slice_names = list(num_users_per_slice.keys())
        self.num_users = num_users_per_slice  # dict: slice_name -> count

        # Generate fixed user positions (distance from BS)
        self.user_distances: Dict[str, np.ndarray] = {}
        for name, count in num_users_per_slice.items():
            # Uniform random within cell, minimum 10m from BS
            distances = self.rng.uniform(10.0, cell_radius_m, size=count)
            self.user_distances[name] = distances

        # Compute fixed path loss per user (in dB)
        self.path_loss_db: Dict[str, np.ndarray] = {}
        for name, distances in self.user_distances.items():
            # Free-space reference at 1m + distance-dependent loss
            fspl_1m = 20 * np.log10(carrier_freq_ghz * 1e9) + 20 * np.log10(4 * np.pi / 3e8)
            pl = fspl_1m + 10 * path_loss_exponent * np.log10(distances)
            self.path_loss_db[name] = pl

        # Current fading state (updated each step)
        self.fading_db: Dict[str, np.ndarray] = {}
        self.snr_db: Dict[str, np.ndarray] = {}
        self._update_fading()

    def _update_fading(self):
        """Generate new Rayleigh fading realization for all users."""
        for name, count in self.num_users.items():
            # Rayleigh fading: |h|^2 is exponential with mean 1
            h_squared = self.rng.exponential(1.0, size=count)
            self.fading_db[name] = 10 * np.log10(h_squared + 1e-10)

            # SNR = Tx power - path loss + fading - noise
            self.snr_db[name] = (
                self.bs_power_dbm
                - self.path_loss_db[name]
                + self.fading_db[name]
                - self.noise_power_dbm
            )

    def step(self):
        """Advance channel state by one time step."""
        self._update_fading()

    def get_snr_linear(self, slice_name: str) -> np.ndarray:
        """Get SNR in linear scale for a slice's users."""
        return 10 ** (self.snr_db[slice_name] / 10.0)

    def get_capacity_mbps(self, slice_name: str, bandwidth_mhz: float) -> np.ndarray:
        """
        Shannon capacity per user: C = B * log2(1 + SNR)
        Returns array of per-user capacities in Mbps.
        """
        snr_lin = self.get_snr_linear(slice_name)
        bw_hz = bandwidth_mhz * 1e6
        capacity_bps = bw_hz * np.log2(1 + snr_lin)
        return capacity_bps / 1e6  # convert to Mbps

    def get_avg_snr_db(self, slice_name: str) -> float:
        """Average SNR across all users in a slice."""
        return float(np.mean(self.snr_db[slice_name]))

    def get_state_vector(self) -> np.ndarray:
        """Flat vector of all channel states for the DRL observation."""
        parts = []
        for name in self.slice_names:
            # Per-slice: mean SNR, min SNR, std SNR (3 values per slice)
            snr = self.snr_db[name]
            parts.extend([np.mean(snr), np.min(snr), np.std(snr)])
        return np.array(parts, dtype=np.float32)
