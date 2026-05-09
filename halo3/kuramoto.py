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


def quantum_potential(theta: jnp.ndarray) -> jnp.ndarray:
    """Bohm's quantum potential from cluster phase distribution.

    Q = -∇²|ψ|/|ψ| approximated as the deviation of each cluster's
    log-amplitude from the mean. Pushes clusters apart when they
    converge in phase space (anti-bunching / diversity maintenance).

    Args:
        theta: (K, n_hidden) cluster phases

    Returns:
        Q: (K, n_hidden) quantum potential force per cluster per dimension
    """
    # ψ amplitude: Gaussian centered on mean phase
    theta_mean = jnp.mean(theta, axis=0, keepdims=True)  # (1, n_hidden)
    deviation = theta - theta_mean                         # (K, n_hidden)

    # |ψ_k| ∝ exp(-½ Σ_d (θ_k_d - θ̄_d)²)
    log_amplitude = -0.5 * jnp.sum(deviation ** 2, axis=-1, keepdims=True)  # (K, 1)

    # Q = -(log|ψ_k| - mean(log|ψ|)) → pushes outliers in, conformists out
    Q_scalar = -(log_amplitude - jnp.mean(log_amplitude))  # (K, 1)

    # Directional quantum force: Q pushes along the deviation direction
    return Q_scalar * deviation  # (K, n_hidden)


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

    # Quantum potential: non-local anti-bunching force
    Q = quantum_potential(state.theta)  # (K, n_hidden)

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
