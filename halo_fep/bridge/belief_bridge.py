# halo_fep/bridge/belief_bridge.py
"""BeliefBridge — maps per-agent belief states to AdS-KG flow conditioning.

Agent beliefs mu_i in R^{n_hidden} (belief logits over eta) are projected to
d_model via w_belief, then distributed back to tokens. The resulting
delta_v is added to the AdS-KG flow vector field in d_model space,
steering what HALO generates toward observations consistent with mu_i.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class BeliefBridge(eqx.Module):
    w_belief: eqx.nn.Linear         # n_hidden -> d_model
    assignment_logits: jnp.ndarray  # (N_agents, N_tok)

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        self.w_belief          = eqx.nn.Linear(cfg.n_hidden, cfg.d_model, key=key)
        self.assignment_logits = jnp.zeros((cfg.n_agents, cfg.n_tokens))

    def __call__(self, mu_i: jnp.ndarray) -> jnp.ndarray:
        """Args:
            mu_i: (N_agents, n_hidden) — agent belief parameters
        Returns:
            delta_v: (N_tok, d_model) — additive flow conditioning
        """
        agent_bias = jax.vmap(self.w_belief)(mu_i)           # (N_agents, d_model)
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)  # (N_agents, N_tok)
        delta_v    = assignment.T @ agent_bias                # (N_tok, d_model)
        return delta_v
