# halo_fep/bridge/obs_bridge.py
"""ObsBridge — maps HALO backbone output to per-agent soft observations.

Each agent has a learned soft assignment over N_tok tokens. The assignment
is row-wise softmax normalized (initialized uniform 1/N_tok). A linear
head maps the pooled d_model embedding to n_obs logits; softmax produces
a differentiable probability vector over observations.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class ObsBridge(eqx.Module):
    assignment_logits: jnp.ndarray  # (N_agents, N_tok) — raw (softmax applied in fwd)
    w_obs: eqx.nn.Linear            # d_model -> n_obs

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        # Initialize assignment uniform (logits = 0 -> softmax = uniform)
        self.assignment_logits = jnp.zeros((cfg.n_agents, cfg.n_tokens))
        self.w_obs = eqx.nn.Linear(cfg.d_model, cfg.n_obs, key=key)

    def _logits(self, h_out: jnp.ndarray) -> jnp.ndarray:
        """Return (N_agents, n_obs) logits — differentiable, before softmax."""
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)  # (N_agents, N_tok)
        h_pooled   = assignment @ h_out                               # (N_agents, d_model)
        return jax.vmap(self.w_obs)(h_pooled)                        # (N_agents, n_obs)

    def __call__(self, h_out: jnp.ndarray) -> jnp.ndarray:
        """Args:
            h_out: (N_tok, d_model) — HALO backbone output
        Returns:
            soft_obs: (N_agents, n_obs) float32 — differentiable observation probabilities
        """
        return jax.nn.softmax(self._logits(h_out), axis=-1)
