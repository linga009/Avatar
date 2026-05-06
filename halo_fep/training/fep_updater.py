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

        # --- Update B (transition P(s'|s,a)) ---
        # Use dominant action from swarm as the action index
        # action_probs shape: (n_actions,) — average over agents
        action_probs = jax.nn.softmax(jnp.mean(carry.swarm_action, axis=0))  # (n_actions,)
        # Transition: outer product of next state and current state beliefs
        # weighted by action probabilities
        # outer_ss: (n_hidden, n_hidden) — self-transition
        outer_ss = jnp.outer(q_eta, q_eta)  # (n_hidden, n_hidden)

        # B is (n_hidden, n_hidden, n_actions) — update each action slice weighted by prob
        log_outer_ss = jnp.log(outer_ss + 1e-8)

        def update_b_slice(log_B, a):
            # a: scalar action index
            action_prob = action_probs[a]          # scalar
            # EMA update for action a slice
            new_slice = alpha * log_B[:, :, a] + (1.0 - alpha) * action_prob * log_outer_ss
            # Normalize over axis 0 (s' dimension)
            new_slice = new_slice - jax.scipy.special.logsumexp(new_slice, axis=0, keepdims=True)
            return log_B, new_slice

        _, b_slices = jax.lax.scan(
            update_b_slice,
            model.gm.log_B,
            jnp.arange(self.cfg.n_actions)
        )
        # b_slices shape: (n_actions, n_hidden, n_hidden) — transpose back to (n_hidden, n_hidden, n_actions)
        new_log_B = jnp.transpose(b_slices, (1, 2, 0))

        new_model = eqx.tree_at(lambda m: m.gm.log_D, model,    new_log_D)
        new_model = eqx.tree_at(lambda m: m.gm.log_A, new_model, new_log_A)
        new_model = eqx.tree_at(lambda m: m.gm.log_B, new_model, new_log_B)
        return new_model
