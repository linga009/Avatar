"""LorentzEmbedding — projects tokens onto Lorentz hyperboloid."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class LorentzEmbedding(eqx.Module):
    x_proj: eqx.nn.Linear
    log_curvature: jnp.ndarray

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        self.x_proj = eqx.nn.Linear(cfg.d_model, cfg.d_boundary - 1, key=key)
        self.log_curvature = jnp.log(jnp.array(cfg.init_curvature))

    @property
    def curvature(self) -> jnp.ndarray:
        # Clamp log_curvature to [-6, 6] so kappa stays in (e^-6, e^6) ≈ (0.002, 403)
        # even after large adafactor updates on the scalar.
        return jnp.exp(jnp.clip(self.log_curvature, -6.0, 6.0))

    def __call__(self, h: jnp.ndarray):
        kappa = self.curvature
        x_spatial = jax.vmap(self.x_proj)(h)
        # Guard the sqrt argument against negative values due to fp rounding.
        x_0 = jnp.sqrt(jnp.maximum(1.0 / kappa + jnp.sum(x_spatial ** 2, axis=-1, keepdims=True), 1e-8))
        x = jnp.concatenate([x_0, x_spatial], axis=-1)
        z = jax.nn.softplus(x_0[:, 0] - 1.0 / jnp.sqrt(kappa))
        return x, z
