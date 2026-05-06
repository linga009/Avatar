# halo_fep/utils.py
"""Shared utility functions for the Persistent Mind system."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from halo_fep.model import HaloFEPCarry, HaloFEPModel


def compute_free_energy(carry: HaloFEPCarry, model: HaloFEPModel) -> jnp.ndarray:
    """Mean KL[Q(eta) || D] over all agents — a scalar measure of surprise.

    This is the intrinsic free energy: how far current beliefs are from the prior.
    Does not require observations (can be called after any step).
    """
    q_eta = jax.nn.softmax(carry.swarm_mu, axis=-1)          # (N_agents, n_hidden)
    log_q = jnp.log(q_eta + 1e-8)                            # (N_agents, n_hidden)
    log_d = jnp.log(model.gm.D + 1e-8)                       # (n_hidden,)
    kl_per_agent = jnp.sum(q_eta * (log_q - log_d[None, :]), axis=-1)  # (N_agents,)
    return jnp.mean(kl_per_agent)
