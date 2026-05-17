"""Bohmian Kuramoto oscillators on n-torus.

Collective inference via pilot-wave-guided synchronization.

Standard Kuramoto uses mean-field coupling: dθ/dt = ω + K·sin(θ_mean - θ).
Bohmian Kuramoto replaces this with a pilot wave from the Hamiltonian's
momentum field plus a quantum potential Q that maintains diversity:

  dθ/dt = ω + pilot_wave + Q(θ) + obs

The pilot wave ∇S is the phase gradient of the collective wave function ψ,
derived from the Hamiltonian backbone's momentum output. Q is the Bohmian
quantum potential: -∇²|ψ|/|ψ|, which pushes clusters apart when they
converge (anti-bunching), preventing the swarm from collapsing.
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config


class KuramotoState(NamedTuple):
    theta: jnp.ndarray       # (K, n_hidden) phases in [0, 2π)
    omega: jnp.ndarray       # (K, n_hidden) natural frequencies
    coupling: float           # scalar K
    key: jnp.ndarray


def init_kuramoto(cfg: Halo3Config, key: jnp.ndarray) -> KuramotoState:
    k1, k2 = jax.random.split(key)
    return KuramotoState(
        theta=jax.random.uniform(k1, (cfg.n_clusters, cfg.n_hidden)) * 2 * jnp.pi,
        omega=jax.random.normal(k2, (cfg.n_clusters, cfg.n_hidden)) * 0.1,
        coupling=cfg.init_coupling,
        key=key,
    )


def quantum_potential(theta: jnp.ndarray, key: jnp.ndarray | None = None) -> jnp.ndarray:
    """Bohm's quantum potential — prevents Kuramoto collapse to r=1.

    Two components:
    1. Standard Q: pushes conformists outward proportional to deviation
    2. Collapse guard: when r > 0.8, injects noise to break synchrony

    Without the collapse guard, aggressive per-tick body learning
    drives all oscillators to r=1.0 (full sync), killing exploration.
    The guard ensures the organism maintains healthy diversity.

    Args:
        theta: (K, n_hidden) cluster phases
        key: optional PRNG key for collapse guard noise

    Returns:
        Q: (K, n_hidden) quantum potential force
    """
    K, n_h = theta.shape
    theta_mean = jnp.mean(theta, axis=0, keepdims=True)  # (1, n_hidden)
    deviation = theta - theta_mean                         # (K, n_hidden)

    # Standard Bohmian Q: pushes based on deviation from mean
    log_amplitude = -0.5 * jnp.sum(deviation ** 2, axis=-1, keepdims=True)
    Q_scalar = -(log_amplitude - jnp.mean(log_amplitude))
    Q_base = Q_scalar * deviation  # (K, n_hidden)

    # Collapse guard: when phases converge, inject repulsive noise
    # The closer to sync (smaller deviation), the stronger the repulsion
    deviation_magnitude = jnp.sqrt(jnp.sum(deviation ** 2, axis=-1, keepdims=True) + 1e-8)
    # Repulsion strength: 1.0 when deviation=0, 0.0 when deviation>1
    repulsion_strength = jnp.exp(-deviation_magnitude)  # (K, 1)

    # Repulsive direction: push each cluster away from mean along a unique direction
    # Use cluster index as deterministic "noise" direction
    cluster_dirs = jnp.sin(jnp.arange(K)[:, None] * 2.397 + jnp.arange(n_h)[None, :] * 1.618)
    Q_repulsion = repulsion_strength * cluster_dirs * 2.0  # (K, n_hidden)

    return Q_base + Q_repulsion


def kuramoto_step(
    state: KuramotoState,
    obs: jnp.ndarray,
    cfg: Halo3Config,
    pilot_wave: jnp.ndarray | None = None,
) -> KuramotoState:
    """One Euler step of Bohmian Kuramoto on n-torus.

    Args:
        state: current KuramotoState
        obs: (K, n_obs) observations as phase kicks
        cfg: config
        pilot_wave: (K, n_hidden) momentum field from Hamiltonian (optional).
                    If None, falls back to classical mean-field coupling.
    """
    n_hid = state.theta.shape[1]

    # Observation drive: project obs to n_hidden dims
    n_obs = obs.shape[1]
    if n_obs >= n_hid:
        obs_drive = obs[:, :n_hid]
    else:
        obs_drive = jnp.pad(obs, ((0, 0), (0, n_hid - n_obs)))

    # Quantum potential: non-local anti-bunching force + collapse guard
    Q = quantum_potential(state.theta)  # (K, n_hidden)
    # Scale Q stronger when r is high to actively resist collapse
    r = jnp.abs(jnp.mean(jnp.exp(1j * state.theta), axis=0))  # (n_hidden,)
    r_mean = jnp.mean(r)
    Q = Q * (1.0 + 3.0 * r_mean)  # Q grows 4× stronger as r→1

    if pilot_wave is not None:
        # Bohmian dynamics: pilot wave (∇S) replaces mean-field coupling
        # pilot_wave is the momentum field from the Hamiltonian, projected to n_hidden
        pw = pilot_wave[:, :n_hid] if pilot_wave.shape[1] >= n_hid else \
            jnp.pad(pilot_wave, ((0, 0), (0, n_hid - pilot_wave.shape[1])))
        coupling_force = state.coupling * pw
    else:
        # Classical fallback: standard Kuramoto mean-field
        theta_mean = jnp.mean(state.theta, axis=0)
        coupling_force = state.coupling * jnp.sin(theta_mean[None, :] - state.theta)

    # Euler step: ω + pilot_wave + Q + obs
    dtheta = state.omega + coupling_force + Q + obs_drive
    new_theta = (state.theta + cfg.kuramoto_dt * dtheta) % (2 * jnp.pi)

    return state._replace(
        theta=new_theta,
        key=jax.random.split(state.key)[0],
    )


def kuramoto_action(state: KuramotoState, n_actions: int) -> jnp.ndarray:
    """Derive action probabilities from phase velocities.

    Returns (K, n_actions) action probabilities.
    """
    theta_mean = jnp.mean(state.theta, axis=0)
    phase_velocity = jnp.sin(theta_mean[None, :] - state.theta)
    return jax.nn.softmax(phase_velocity[:, :n_actions], axis=-1)


def order_parameter(theta: jnp.ndarray) -> jnp.ndarray:
    """Kuramoto order parameter r ∈ [0,1] per hidden dimension.

    r=1 means full synchronization, r=0 means uniform spread.
    """
    return jnp.abs(jnp.mean(jnp.exp(1j * theta), axis=0))


def dual_order_parameters(theta: jnp.ndarray) -> tuple:
    """Split phases into analytical/creative populations and measure body tension.

    Splits n_hidden phases down the middle:
      - Analytical ([:n_h//2]): precision, convergence, evaluation
      - Creative   ([n_h//2:]): divergence, exploration, generation

    Body tension = |r_analytical - r_creative| ∈ [0, 1]
      0.0 = both populations agree on the pattern (clear signal)
      1.0 = populations completely split (organism is of two minds)

    Returns:
        (r_analytical, r_creative, body_tension) — all Python floats
    """
    n_h = theta.shape[1]
    mid = n_h // 2
    r_a = jnp.mean(order_parameter(theta[:, :mid]))
    r_c = jnp.mean(order_parameter(theta[:, mid:]))
    body_tension = jnp.abs(r_a - r_c)
    return r_a, r_c, body_tension
