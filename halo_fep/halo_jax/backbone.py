# halo_fep/halo_jax/backbone.py
"""HALOBackbone — 8-layer [S,S,S,H,S,S,S,H] stack in JAX/equinox.

Each layer: LayerNorm -> core (SSM or HoloAttn) -> residual -> LayerNorm -> FFN -> residual.

The layer_types tuple is static so JAX can trace through the for-loop
without dynamic branching at JIT time.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig
from halo_fep.halo_jax.simple_ssm import SimpleSSM
from halo_fep.halo_jax.holo_attention import HoloAttention


class _FFN(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, d_model: int, d_ff: int, *, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(d_model, d_ff, key=k1)
        self.fc2 = eqx.nn.Linear(d_ff, d_model, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.fc2(jax.nn.gelu(self.fc1(x)))


class HALOBackbone(eqx.Module):
    layers: list       # [SimpleSSM | HoloAttention] * n_layers
    norms1: list       # LayerNorm before core layer
    norms2: list       # LayerNorm before FFN
    ffns: list         # _FFN per layer
    layer_types: tuple = eqx.field(static=True)  # ("S","S","S","H","S","S","S","H")

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        self.layer_types = ("S", "S", "S", "H", "S", "S", "S", "H")
        assert len(self.layer_types) == cfg.n_layers

        keys = jax.random.split(key, cfg.n_layers * 3)
        self.layers = []
        self.norms1 = []
        self.norms2 = []
        self.ffns   = []

        for i, lt in enumerate(self.layer_types):
            k_core, k_ffn = keys[i * 3], keys[i * 3 + 1]
            if lt == "S":
                self.layers.append(SimpleSSM(cfg, k_core))
            else:
                self.layers.append(HoloAttention(cfg, k_core))
            self.norms1.append(eqx.nn.LayerNorm(cfg.d_model))
            self.norms2.append(eqx.nn.LayerNorm(cfg.d_model))
            self.ffns.append(_FFN(cfg.d_model, cfg.d_ff, key=k_ffn))

    def __call__(
        self,
        h: jnp.ndarray,                          # (N_tok, d_model)
        x: jnp.ndarray,                           # (N_tok, d_boundary)
        z: jnp.ndarray,                           # (N_tok, 1)
        delta_x: Optional[jnp.ndarray] = None,   # (N_tok, d_boundary) action bias
    ) -> jnp.ndarray:                             # (N_tok, d_model)
        for layer, norm1, norm2, ffn, lt in zip(
            self.layers, self.norms1, self.norms2, self.ffns, self.layer_types
        ):
            # Pre-norm + core
            h_n = jax.vmap(norm1)(h)
            if lt == "S":
                h_core = layer(h_n)
            else:
                h_core = layer(h_n, x, z, delta_x=delta_x)
            h = h + h_core
            # Pre-norm + FFN
            h = h + jax.vmap(ffn)(jax.vmap(norm2)(h))
        return h
