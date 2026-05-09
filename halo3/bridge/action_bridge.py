"""ActionBridge — Kuramoto actions -> boundary bias."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class ActionBridge(eqx.Module):
    w_action: eqx.nn.Linear
    assignment_logits: jnp.ndarray

    def __init__(self, cfg: Halo3Config, key):
        self.w_action = eqx.nn.Linear(cfg.n_actions, cfg.d_boundary, key=key)
        self.assignment_logits = jnp.zeros((cfg.n_clusters, cfg.n_tokens))

    def __call__(self, a_i):
        agent_bias = jax.vmap(self.w_action)(a_i)
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)
        return assignment.T @ agent_bias
