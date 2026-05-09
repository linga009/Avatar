"""Halo3Backbone — reversible backbone with MERA-FFN and S7 SSM / SharedHoloAttention."""
from __future__ import annotations
import dataclasses
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config
from halo3.ssm_s7 import SelectiveSSM
from halo3.holo_attention_shared import SharedHoloAttention, LoRAAdapter
from halo3.mera_ffn import MERAFFN


class _BackboneLayer(eqx.Module):
    """Single pre-norm layer: core (SSM or SharedHoloAttention) + MERA-FFN."""
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    core: eqx.Module          # SelectiveSSM or SharedHoloAttention
    ffn: MERAFFN
    layer_type: str = eqx.field(static=True)  # "S" or "H"

    def __init__(
        self,
        half_cfg: Halo3Config,
        layer_type: str,
        key: jnp.ndarray,
    ) -> None:
        k_core, k_ffn = jax.random.split(key)
        d = half_cfg.d_model
        self.norm1 = eqx.nn.LayerNorm(d)
        self.norm2 = eqx.nn.LayerNorm(d)
        self.layer_type = layer_type
        if layer_type == "S":
            self.core = SelectiveSSM(half_cfg, k_core)
        else:
            self.core = SharedHoloAttention(half_cfg, k_core)
        self.ffn = MERAFFN(half_cfg, k_ffn)

    def __call__(
        self,
        h: jnp.ndarray,      # (N_tok, d_half)
        x: jnp.ndarray,      # (N_tok, d_boundary) — Lorentz positions
        z: jnp.ndarray,      # (N_tok,)
        lora=None,
    ) -> jnp.ndarray:
        # Pre-norm + core
        h_n = jax.vmap(self.norm1)(h)
        if self.layer_type == "S":
            h_core = self.core(h_n)
        else:
            h_core = self.core(h_n, x, z, lora=lora)
        h = h + h_core
        # Pre-norm + MERA-FFN (operates on individual vectors)
        h = h + jax.vmap(self.ffn)(jax.vmap(self.norm2)(h))
        return h


class Halo3Backbone(eqx.Module):
    """Reversible backbone using half-width split.

    Operates on d_half = d_model // 2 streams (F and G), alternating
    reversible coupling to allow gradient checkpointing in the future.
    Shared LoRA adapter is provided to attention layers.

    Call signature: __call__(h, q_data, z) → h_out
      h:       (n_tokens, d_model)
      q_data:  (n_tokens, d_boundary) from LorentzEmbedding
      z:       (n_tokens,)
    """
    layers: list          # list of _BackboneLayer, each at d_half width
    lora: LoRAAdapter
    merge_proj: eqx.nn.Linear   # 2*d_half → d_model (combine F+G streams)
    split_proj: eqx.nn.Linear   # d_model → 2*d_half (split into F+G streams)
    layer_types: tuple = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        d_half = cfg.d_model // 2
        n_heads_half = cfg.n_heads // 2
        # Build a half-width config for all sub-layers
        half_cfg = dataclasses.replace(
            cfg,
            d_model=d_half,
            n_heads=n_heads_half,
        )

        # Expand the layer pattern to match n_layers
        repeats = cfg.n_layers // len(cfg.layer_pattern)
        full_pattern = cfg.layer_pattern * repeats
        self.layer_types = tuple(full_pattern)

        n_layers = len(self.layer_types)
        keys = jax.random.split(key, n_layers + 3)

        self.layers = [
            _BackboneLayer(half_cfg, lt, keys[i])
            for i, lt in enumerate(self.layer_types)
        ]
        self.lora = LoRAAdapter(half_cfg, keys[n_layers])
        self.split_proj = eqx.nn.Linear(cfg.d_model, 2 * d_half, key=keys[n_layers + 1])
        self.merge_proj = eqx.nn.Linear(2 * d_half, cfg.d_model, key=keys[n_layers + 2])

    def __call__(
        self,
        h: jnp.ndarray,       # (n_tokens, d_model)
        q_data: jnp.ndarray,  # (n_tokens, d_boundary)
        z: jnp.ndarray,       # (n_tokens,)
    ) -> jnp.ndarray:         # (n_tokens, d_model)
        # Split into two half-width streams
        split = jax.vmap(self.split_proj)(h)          # (n_tokens, 2*d_half)
        d_half = split.shape[-1] // 2
        f = split[:, :d_half]   # (n_tokens, d_half)
        g = split[:, d_half:]   # (n_tokens, d_half)

        # Reversible coupling: alternately update F from G and G from F
        for i, layer in enumerate(self.layers):
            lora_arg = self.lora if layer.layer_type == "H" else None
            if i % 2 == 0:
                # Update F using G as context (pass G through this layer)
                delta = layer(g, q_data, z, lora=lora_arg)
                f = f + delta
            else:
                # Update G using F as context
                delta = layer(f, q_data, z, lora=lora_arg)
                g = g + delta

        # Merge streams back to d_model
        combined = jnp.concatenate([f, g], axis=-1)  # (n_tokens, 2*d_half)
        return jax.vmap(self.merge_proj)(combined)    # (n_tokens, d_model)
