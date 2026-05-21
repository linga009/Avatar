# halo_fep/training/fep_updater.py
"""Online Bayesian updates to FEP generative model matrices A, B, D.

After each subconscious tick the heartbeat loop calls ``FEPUpdater.update``
with the current carry **and** the ``soft_obs`` array produced by
``ObsBridge``.  Using the real observation distribution (rather than the
action-proxy used in the original code) ensures that the likelihood matrix
**A** tracks what the agent *sees*, not what it *does*.

Mathematics
-----------
Updates are log-space Exponential Moving Averages, approximating Dirichlet
posterior updates:

    log_D_new = α · log_D + (1-α) · log( E_agents[softmax(swarm_mu)] )
    log_A_new = α · log_A + (1-α) · log( outer(soft_obs_mean, q_eta_mean) )
    log_B_new[a] = α · log_B[a] + (1-α) · p(a) · log( outer(q_eta, q_eta) )

α = 0.99 means ~100 ticks to fully update priors from scratch.

Bug fix summary
---------------
Previously the A-matrix update used ``carry.swarm_action`` as a proxy for
``soft_obs``, silently conflating the agent's *action preferences* with its
*perceptual observations*.  This caused A to converge toward representing
action biases rather than observation likelihoods, breaking the FEP loop.
The fix is to accept ``soft_obs`` as an explicit (N_agents, n_obs) argument.
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
    """Online Bayesian updates to the FEP generative model.

    Attributes
    ----------
    cfg   : System configuration.
    alpha : EMA retention coefficient.
    """

    def __init__(self, cfg: HaloFEPConfig, alpha: float = _ALPHA) -> None:
        self.cfg   = cfg
        self.alpha = alpha

    def update(
        self,
        model: HaloFEPModel,
        carry: HaloFEPCarry,
        episode: Episode,
        soft_obs: jnp.ndarray,
    ) -> HaloFEPModel:
        """Update log_D, log_A, and log_B from the current tick's data.

        Parameters
        ----------
        model    : Current model (will not be mutated; returns a new model).
        carry    : Current HaloFEPCarry (provides swarm_mu, swarm_action).
        episode  : Stored episode (not used directly; here for future use).
        soft_obs : (N_agents, n_obs) float32 — real observation probabilities
                   from ``ObsBridge.__call__``.  **Must not be substituted
                   with swarm_action.**

        Returns
        -------
        HaloFEPModel with updated gm.log_D, gm.log_A, gm.log_B.
        """
        alpha = self.alpha

        # ------------------------------------------------------------------
        # Posterior belief: mean over agents → (n_hidden,)
        # ------------------------------------------------------------------
        q_eta = jax.nn.softmax(
            jnp.mean(carry.swarm_mu, axis=0)
        )                                                   # (n_hidden,)

        # ------------------------------------------------------------------
        # Update D — prior over hidden states
        # ------------------------------------------------------------------
        log_q     = jnp.log(q_eta + 1e-8)                 # (n_hidden,)
        new_log_D = alpha * model.gm.log_D + (1.0 - alpha) * log_q
        # Normalise so softmax(log_D) remains a proper distribution
        new_log_D = new_log_D - jax.scipy.special.logsumexp(new_log_D)

        # ------------------------------------------------------------------
        # Update A — observation likelihood P(obs | hidden)
        # Use the REAL soft_obs from ObsBridge, NOT the action proxy.
        # Mean over agents: (n_agents, n_obs) → (n_obs,)
        # ------------------------------------------------------------------
        soft_obs_mean = jnp.mean(soft_obs, axis=0)         # (n_obs,)
        outer         = jnp.outer(soft_obs_mean, q_eta)    # (n_obs, n_hidden)
        log_outer     = jnp.log(outer + 1e-8)
        new_log_A     = alpha * model.gm.log_A + (1.0 - alpha) * log_outer
        # Normalise each column (per hidden state) so A[:,j] is a distribution
        new_log_A = new_log_A - jax.scipy.special.logsumexp(
            new_log_A, axis=0, keepdims=True
        )

        # ------------------------------------------------------------------
        # Update B — transition P(s' | s, a)
        # Dominant action from swarm weighted by action probabilities.
        # ------------------------------------------------------------------
        action_probs = jax.nn.softmax(
            jnp.mean(carry.swarm_action, axis=0)
        )                                                   # (n_actions,)
        outer_ss     = jnp.outer(q_eta, q_eta)             # (n_hidden, n_hidden)
        log_outer_ss = jnp.log(outer_ss + 1e-8)

        def update_b_slice(log_B, a):
            """EMA update for the B-matrix slice corresponding to action a."""
            action_prob = action_probs[a]
            new_slice = (
                alpha * log_B[:, :, a]
                + (1.0 - alpha) * action_prob * log_outer_ss
            )
            # Normalise over s' dimension (axis 0)
            new_slice = new_slice - jax.scipy.special.logsumexp(
                new_slice, axis=0, keepdims=True
            )
            return log_B, new_slice

        _, b_slices = jax.lax.scan(
            update_b_slice,
            model.gm.log_B,
            jnp.arange(self.cfg.n_actions),
        )
        # b_slices: (n_actions, n_hidden, n_hidden) → transpose to (n_hidden, n_hidden, n_actions)
        new_log_B = jnp.transpose(b_slices, (1, 2, 0))

        # ------------------------------------------------------------------
        # Apply updates via eqx.tree_at (returns a new model, no mutation)
        # ------------------------------------------------------------------
        new_model = eqx.tree_at(lambda m: m.gm.log_D, model,     new_log_D)
        new_model = eqx.tree_at(lambda m: m.gm.log_A, new_model, new_log_A)
        new_model = eqx.tree_at(lambda m: m.gm.log_B, new_model, new_log_B)
        return new_model
