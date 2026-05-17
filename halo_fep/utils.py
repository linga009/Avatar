# halo_fep/utils.py
"""Shared utility functions for the Persistent Mind system.

Functions
---------
compute_free_energy
    Scalar KL-divergence measure of the swarm's current surprise level.
    Used by the heartbeat loop to decide whether to trigger the LLM wake cycle.

free_energy_stats
    Extended statistics (mean, std, min, max, median) for calibration and
    monitoring.  Useful for deciding a good ``wake_threshold`` value.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from halo_fep.model import HaloFEPCarry, HaloFEPModel


def compute_free_energy(carry: HaloFEPCarry, model: HaloFEPModel) -> jnp.ndarray:
    """Compute mean KL[Q(η) || D] over all agents — a scalar surprise measure.

    This is the *intrinsic* free energy: how far the current swarm beliefs are
    from the prior D.  It does not require observations and can be called after
    any ``halo_fep_step``.

    Mathematical form
    -----------------
    For each agent i with posterior q_i = softmax(swarm_mu[i]):

        F_i = KL(q_i || D) = Σ_s q_i(s) · [log q_i(s) - log D(s)]

    The scalar returned is the mean over all N_agents.

    Parameters
    ----------
    carry : Current cognitive state (provides swarm_mu).
    model : Current model (provides gm.D — the prior).

    Returns
    -------
    Scalar float32 JAX array ≥ 0.  Zero only when all agents' posteriors equal D.
    """
    q_eta       = jax.nn.softmax(carry.swarm_mu, axis=-1)         # (N_agents, n_hidden)
    log_q       = jnp.log(q_eta + 1e-8)                           # (N_agents, n_hidden)
    log_d       = jnp.log(model.gm.D + 1e-8)                      # (n_hidden,)
    kl_per_agent = jnp.sum(q_eta * (log_q - log_d[None, :]), axis=-1)  # (N_agents,)
    return jnp.mean(kl_per_agent)


class FreeEnergyStats(NamedTuple):
    """Statistics for the per-agent free-energy distribution.

    Attributes
    ----------
    mean   : Mean KL over agents (same as compute_free_energy).
    std    : Standard deviation over agents.
    min    : Minimum KL (most certain agent).
    max    : Maximum KL (most surprised agent).
    median : Median KL.
    """
    mean:   float
    std:    float
    min:    float
    max:    float
    median: float


def free_energy_stats(carry: HaloFEPCarry, model: HaloFEPModel) -> FreeEnergyStats:
    """Compute extended free-energy statistics for calibration and monitoring.

    Use these values to calibrate ``wake_threshold``.  For example, if the
    median FE during normal operation is 0.3 nats and the max is 1.5 nats,
    setting ``wake_threshold = 2.0`` will trigger the LLM only on genuinely
    anomalous observations.

    Parameters
    ----------
    carry : Current cognitive state.
    model : Current model.

    Returns
    -------
    FreeEnergyStats NamedTuple with mean, std, min, max, median (all in nats).
    """
    q_eta        = jax.nn.softmax(carry.swarm_mu, axis=-1)
    log_q        = jnp.log(q_eta + 1e-8)
    log_d        = jnp.log(model.gm.D + 1e-8)
    kl_per_agent = jnp.sum(q_eta * (log_q - log_d[None, :]), axis=-1)   # (N_agents,)

    return FreeEnergyStats(
        mean   = float(jnp.mean(kl_per_agent)),
        std    = float(jnp.std(kl_per_agent)),
        min    = float(jnp.min(kl_per_agent)),
        max    = float(jnp.max(kl_per_agent)),
        median = float(jnp.median(kl_per_agent)),
    )
