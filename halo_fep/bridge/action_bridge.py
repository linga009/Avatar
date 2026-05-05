# halo_fep/bridge/action_bridge.py
"""ActionBridge — maps per-agent action distributions to K_Delta boundary bias.

Agent actions a_i in R^{n_actions} (policy probabilities, softmax-normalized)
are projected to d_boundary via w_action, then distributed back to tokens
via a learned token assignment.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class ActionBridge(eqx.Module):
    w_action: eqx.nn.Linear         # n_actions -> d_boundary
    assignment_logits: jnp.ndarray  # (N_agents, N_tok) — own assignment

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        self.w_action          = eqx.nn.Linear(cfg.n_actions, cfg.d_boundary, key=key)
        self.assignment_logits = jnp.zeros((cfg.n_agents, cfg.n_tokens))

    def __call__(self, a_i: jnp.ndarray) -> jnp.ndarray:
        """Args:
            a_i: (N_agents, n_actions) — action probability vectors
        Returns:
            delta_x: (N_tok, d_boundary) — additive boundary bias for K_Delta
        """
        agent_bias = jax.vmap(self.w_action)(a_i)            # (N_agents, d_boundary)
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)  # (N_agents, N_tok)
        # Distribute agent biases back to tokens: (N_tok, d_boundary)
        delta_x    = assignment.T @ agent_bias
        return delta_x
