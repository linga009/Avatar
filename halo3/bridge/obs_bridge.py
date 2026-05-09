"""ObsBridge — backbone output -> per-cluster observations for Kuramoto."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class ObsBridge(eqx.Module):
    assignment_logits: jnp.ndarray  # (K, n_tokens)
    w_obs: eqx.nn.Linear            # d_model -> n_obs

    def __init__(self, cfg: Halo3Config, key):
        self.assignment_logits = jnp.zeros((cfg.n_clusters, cfg.n_tokens))
        self.w_obs = eqx.nn.Linear(cfg.d_model, cfg.n_obs, key=key)

    def __call__(self, h_out):
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)
        h_pooled = assignment @ h_out
        return jax.nn.softmax(jax.vmap(self.w_obs)(h_pooled), axis=-1)
