"""MetaLayer — modifies Kuramoto coupling K and frequencies ω."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


@dataclass
class MetaCarry:
    ring_buffer: jnp.ndarray   # (meta_k, n_hidden) recent mean phases
    tick_count: int


class MetaLayer(eqx.Module):
    """Fires every meta_k ticks, outputs coupling and frequency adjustments."""
    w_coupling: eqx.nn.Linear   # n_hidden → 1 (coupling adjustment)
    w_omega: eqx.nn.Linear      # n_hidden → n_hidden (frequency adjustment)
    _meta_k: int = eqx.field(static=True)
    _n_hidden: int = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key):
        k1, k2 = jax.random.split(key)
        self.w_coupling = eqx.nn.Linear(cfg.n_hidden, 1, key=k1)
        self.w_omega = eqx.nn.Linear(cfg.n_hidden, cfg.n_hidden, key=k2)
        self._meta_k = cfg.meta_k
        self._n_hidden = cfg.n_hidden

    def init_carry(self) -> MetaCarry:
        return MetaCarry(
            ring_buffer=jnp.zeros((self._meta_k, self._n_hidden)),
            tick_count=0,
        )

    def step(self, carry, mean_phase, order_r):
        """Process one tick. Returns (new_carry, (delta_K, delta_omega) or None)."""
        idx = carry.tick_count % self._meta_k
        new_ring = carry.ring_buffer.at[idx].set(mean_phase)
        new_tick = carry.tick_count + 1

        if new_tick % self._meta_k != 0:
            return MetaCarry(ring_buffer=new_ring, tick_count=new_tick), None

        # Meta-step fires
        ring_mean = jnp.mean(new_ring, axis=0)  # (n_hidden,)
        delta_K = float(jnp.tanh(self.w_coupling(ring_mean)).squeeze())
        delta_omega = jnp.tanh(self.w_omega(ring_mean)) * 0.01  # small adjustments

        return MetaCarry(ring_buffer=new_ring, tick_count=new_tick), (delta_K, delta_omega)
