"""BeliefBridge — Kuramoto phases -> d_model conditioning via sin/cos."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class BeliefBridge(eqx.Module):
    w_belief: eqx.nn.Linear         # 2*n_hidden -> d_model
    assignment_logits: jnp.ndarray   # (K, n_tokens)

    def __init__(self, cfg: Halo3Config, key):
        self.w_belief = eqx.nn.Linear(2 * cfg.n_hidden, cfg.d_model, key=key)
        self.assignment_logits = jnp.zeros((cfg.n_clusters, cfg.n_tokens))

    def __call__(self, theta):
        """theta: (K, n_hidden) phases -> (n_tokens, d_model) conditioning."""
        encoded = jnp.concatenate([jnp.sin(theta), jnp.cos(theta)], axis=-1)
        agent_bias = jax.vmap(self.w_belief)(encoded)
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)
        return assignment.T @ agent_bias
