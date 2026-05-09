"""SharedHoloAttention — Zamba2-style shared AdS/CFT attention with LoRA."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config
from halo3.lorentz_ops import lorentz_distance

EPS = 1e-6


class LoRAAdapter(eqx.Module):
    A_v: jnp.ndarray
    B_v: jnp.ndarray
    A_out: jnp.ndarray
    B_out: jnp.ndarray

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        r = cfg.lora_rank
        hd = cfg.n_heads * cfg.d_head
        self.A_v = jax.random.normal(k1, (cfg.d_model, r)) * 0.01
        self.B_v = jax.random.normal(k2, (r, hd)) * 0.01
        self.A_out = jax.random.normal(k3, (hd, r)) * 0.01
        self.B_out = jax.random.normal(k4, (r, cfg.d_model)) * 0.01


class SharedHoloAttention(eqx.Module):
    log_delta: jnp.ndarray
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.log_delta = jnp.zeros(cfg.n_heads)
        self.v_proj = eqx.nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, key=k1)
        self.out_proj = eqx.nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, key=k2)
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

    def __call__(self, h, x, z, delta_x=None, lora=None):
        N_tok = h.shape[0]
        V = jax.vmap(self.v_proj)(h)
        if lora is not None:
            V = V + h @ lora.A_v @ lora.B_v
        V_heads = V.reshape(N_tok, self.n_heads, self.d_head).transpose(1, 0, 2)

        x_b = x if delta_x is None else x + delta_x
        d_geo = jax.vmap(
            lambda xi: jax.vmap(lambda xj: lorentz_distance(xi, xj, 1.0))(x_b)
        )(x_b)
        base = 1.0 / (jnp.cosh(d_geo) + 1.0 + EPS)

        def head_attn(log_d, v_h):
            K = base ** jnp.exp(log_d)
            A = jax.nn.softmax(K, axis=-1)
            return A @ v_h

        head_outs = jax.vmap(head_attn)(self.log_delta, V_heads)
        concat = head_outs.transpose(1, 0, 2).reshape(N_tok, self.n_heads * self.d_head)
        out = jax.vmap(self.out_proj)(concat)
        if lora is not None:
            out = out + concat @ lora.A_out @ lora.B_out
        return out
