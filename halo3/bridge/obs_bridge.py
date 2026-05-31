"""ObsBridge — backbone output -> per-cluster phase observations for Kuramoto."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class ObsBridge(eqx.Module):
    assignment_logits: jnp.ndarray  # (K, n_tokens)
    w_obs: eqx.nn.Linear            # d_model -> n_obs * 2 (sin/cos pairs for atan2)

    def __init__(self, cfg: Halo3Config, key):
        self.assignment_logits = jnp.zeros((cfg.n_clusters, cfg.n_tokens))
        self.w_obs = eqx.nn.Linear(cfg.d_model, cfg.n_obs * 2, key=key)

    def __call__(self, h_out):
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)
        h_pooled = assignment @ h_out  # (K, d_model)
        raw = jax.vmap(self.w_obs)(h_pooled)  # (K, n_obs * 2)
        # Phase projection: atan2(sin_part, cos_part) -> [-pi, pi]
        n_obs = raw.shape[1] // 2
        sin_part = raw[:, :n_obs]
        cos_part = raw[:, n_obs:]
        return jnp.arctan2(sin_part, cos_part)  # (K, n_obs)
