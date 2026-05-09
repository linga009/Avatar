"""HomeostaticRegulator — uses order parameter r for explore/exploit."""
from __future__ import annotations
import jax.numpy as jnp
from halo3.config import Halo3Config


class HomeostaticRegulator:
    def __init__(self, cfg: Halo3Config):
        self.cfg = cfg
        self.r_ema: float = 0.5  # running average of order parameter

    def update(self, order_r: jnp.ndarray) -> tuple[float, str]:
        """Update with current order parameter. Returns (r_mean, mode)."""
        r_mean = float(jnp.mean(order_r))
        self.r_ema = 0.99 * self.r_ema + 0.01 * r_mean

        if self.r_ema < self.cfg.homeo_sync_threshold:
            return r_mean, "explore"  # low sync → reduce coupling → diversify
        else:
            return r_mean, "exploit"  # high sync → increase coupling → focus
