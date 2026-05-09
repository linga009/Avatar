"""SelectiveSSM — S7-style input-dependent SSM with sequential scan."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class SelectiveSSM(eqx.Module):
    A: jnp.ndarray
    D: jnp.ndarray
    W_gate: eqx.nn.Linear
    W_dt: eqx.nn.Linear
    W_B: eqx.nn.Linear
    W_C: eqx.nn.Linear

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.A = jnp.full((cfg.d_state,), -1.0)
        self.D = jnp.ones(cfg.d_model)
        self.W_gate = eqx.nn.Linear(cfg.d_model, cfg.d_state, use_bias=False, key=k1)
        self.W_dt = eqx.nn.Linear(cfg.d_model, cfg.d_state, use_bias=False, key=k2)
        self.W_B = eqx.nn.Linear(cfg.d_model, cfg.d_state, use_bias=False, key=k3)
        self.W_C = eqx.nn.Linear(cfg.d_state, cfg.d_model, use_bias=False, key=k4)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __call__(self, xs: jnp.ndarray) -> jnp.ndarray:
        gates = jax.nn.sigmoid(jax.vmap(self.W_gate)(xs))
        dt = jax.nn.softplus(jax.vmap(self.W_dt)(xs))
        B_t = jax.vmap(self.W_B)(xs)
        A_bar = jnp.exp(self.A[None, :] * dt)
        gated_A = gates * A_bar
        gated_B = (1.0 - gates) * B_t

        def step(h, inputs):
            a, b = inputs
            h_new = a * h + b
            return h_new, h_new

        _, hs = jax.lax.scan(step, jnp.zeros_like(gated_A[0]), (gated_A, gated_B))
        ys = jax.vmap(self.W_C)(hs) + self.D[None, :] * xs
        return ys
