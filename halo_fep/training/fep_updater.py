# halo_fep/training/fep_updater.py
"""Online Bayesian updates to FEP generative model matrices.

After each subconscious tick, update A/B/D via exponential moving average
in log-space. This approximates a Dirichlet posterior update.

  log_D_new = alpha * log_D + (1-alpha) * log(q_eta_mean)
  log_A_new = alpha * log_A + (1-alpha) * log(outer(soft_obs_mean, q_eta_mean))

alpha=0.99 means ~100 episodes to fully update priors.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPCarry, HaloFEPModel
from halo_fep.memory.schema import Episode

_ALPHA = 0.99   # EMA decay (slower = more conservative)


class FEPUpdater:
    def __init__(self, cfg: HaloFEPConfig, alpha: float = _ALPHA) -> None:
        self.cfg   = cfg
        self.alpha = alpha

    def update(
        self,
        model: HaloFEPModel,
        carry: HaloFEPCarry,
        episode: Episode,
    ) -> HaloFEPModel:
        """Update log_D and log_A from carry beliefs. Returns new model."""
        alpha = self.alpha

        # Posterior belief: mean over agents -> (n_hidden,)
        q_eta = jax.nn.softmax(
            jnp.mean(carry.swarm_mu, axis=0)
        )                                                   # (n_hidden,)

        # --- Update D (prior over hidden states) ---
        log_q   = jnp.log(q_eta + 1e-8)                   # (n_hidden,)
        new_log_D = alpha * model.gm.log_D + (1.0 - alpha) * log_q
        # Normalize so softmax(log_D) remains a proper distribution
        new_log_D = new_log_D - jax.scipy.special.logsumexp(new_log_D)

        # --- Update A (likelihood P(obs | hidden)) ---
        # Use mean soft_obs from ObsBridge output (via mean swarm_action as proxy)
        # We approximate soft_obs as the current action distribution (n_actions,)
        # reshaped to (n_obs,) if n_obs == n_actions, else use uniform
        if self.cfg.n_obs == self.cfg.n_actions:
            soft_obs = jnp.mean(carry.swarm_action, axis=0)  # (n_obs,)
        else:
            soft_obs = jnp.ones(self.cfg.n_obs) / self.cfg.n_obs

        # outer product: (n_obs, n_hidden)
        outer    = jnp.outer(soft_obs, q_eta)              # (n_obs, n_hidden)
        log_outer = jnp.log(outer + 1e-8)
        new_log_A = alpha * model.gm.log_A + (1.0 - alpha) * log_outer
        # Normalize each column (hidden state) so A[:,j] is a distribution
        new_log_A = new_log_A - jax.scipy.special.logsumexp(new_log_A, axis=0, keepdims=True)

        new_model = eqx.tree_at(lambda m: m.gm.log_D, model,    new_log_D)
        new_model = eqx.tree_at(lambda m: m.gm.log_A, new_model, new_log_A)
        return new_model
