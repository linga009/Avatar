"""Lorentz hyperboloid operations — pure functions, no parameters."""
from __future__ import annotations
import jax.numpy as jnp


def lorentz_inner(x: jnp.ndarray, y: jnp.ndarray, kappa: float) -> jnp.ndarray:
    return -x[0] * y[0] + jnp.dot(x[1:], y[1:])


def lorentz_distance(x: jnp.ndarray, y: jnp.ndarray, kappa: float) -> jnp.ndarray:
    inner = -kappa * lorentz_inner(x, y, kappa)
    # jnp.where keeps the forward value correct (arccosh(1)=0 when inner=1),
    # but evaluates arccosh at a safe point (inner_safe >= 1+eps) so the
    # backward pass never hits the arccosh singularity at x=1.
    inner_safe = jnp.maximum(inner, 1.0 + 1e-7)
    return jnp.where(inner <= 1.0, 0.0, jnp.arccosh(inner_safe)) / jnp.sqrt(kappa)


def exp_map(x: jnp.ndarray, v: jnp.ndarray, kappa: float) -> jnp.ndarray:
    v_norm = jnp.sqrt(jnp.clip(kappa * lorentz_inner(v, v, kappa), 1e-8, None))
    return jnp.cosh(v_norm) * x + jnp.sinh(v_norm) * v / (v_norm + 1e-8)


def log_map(x: jnp.ndarray, y: jnp.ndarray, kappa: float) -> jnp.ndarray:
    alpha = -kappa * lorentz_inner(x, y, kappa)
    alpha = jnp.clip(alpha, 1.0 + 1e-7, None)
    coeff = jnp.arccosh(alpha) / jnp.sqrt(jnp.clip(alpha ** 2 - 1.0, 1e-8, None))
    return coeff * (y - alpha * x)
