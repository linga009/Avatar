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
    mid = cfg.n_hidden // 2

    # Dual-population natural frequencies (v3.4):
    #   Analytical [:mid] — low std, slow oscillators that want to synchronize.
    #                        Converges naturally → high r_a = precision/evaluation mode.
    #   Creative   [mid:] — high std, fast oscillators that stay spread.
    #                        Resists sync naturally → low r_c = exploration/generation mode.
    # body_tension = |r_a - r_c| now measures genuine functional divergence,
    # not arbitrary noise from a uniform population.
    omega_analytical = jax.random.normal(k2, (cfg.n_clusters, mid)) * 0.03
    omega_creative   = jax.random.normal(jax.random.fold_in(k2, 1), (cfg.n_clusters, cfg.n_hidden - mid)) * 0.8
    omega = jnp.concatenate([omega_analytical, omega_creative], axis=1)

    return KuramotoState(
        theta=jax.random.uniform(k1, (cfg.n_clusters, cfg.n_hidden)) * 2 * jnp.pi,
        omega=omega,
        coupling=cfg.init_coupling,
        key=key,
    )


def quantum_potential(theta: jnp.ndarray, key: jnp.ndarray | None = None) -> jnp.ndarray:
    """Bohmian quantum force on the n-torus.

    Computes F_Q = -Q_scale * dQ/dtheta where Q = -nabla^2 sqrt(rho) / sqrt(rho).
    The canonical ½ prefactor is absorbed into Q_scale (0.01), which was
    empirically tuned to balance anti-bunching with Kuramoto coupling at
    the critical region (r ~ 0.5, K ~ 0.3).

    Phase density rho is estimated via von Mises kernel smoothing (the
    circular analogue of Gaussian KDE), which correctly handles the
    periodic topology of [0, 2pi).

    Anti-bunching emerges naturally from wavefunction curvature:
    - Where phases cluster, amplitude curvature is negative, Q is high
    - The gradient of Q pushes oscillators away from density peaks
    - At perfect sync, Q is constant (all phases at the same peak),
      so the gradient (force) is exactly zero

    Q_scale (default 0.01) is the Avatar equivalent of hbar^2/2m in
    real Bohmian mechanics — it controls how strongly quantum anti-bunching
    acts relative to classical Kuramoto coupling. The raw autodiff force
    is O(10-100); scaling by 0.01 brings it into balance with coupling
    force O(0.1-1) at the critical region (r ~ 0.5, K ~ 0.3).

    Args:
        theta: (K, n_hidden) cluster phases in [0, 2pi)
        key: unused (kept for interface compatibility)

    Returns:
        F_Q: (K, n_hidden) quantum force (phase velocity contribution)
    """
    K, n_h = theta.shape
    kappa = 2.0   # von Mises concentration (kernel bandwidth)
    Q_scale = 0.01  # absorbs ½ from canonical Q = -(½)∇²√ρ/√ρ; tuned for K~0.3 balance

    def _Q_total(theta_flat: jnp.ndarray) -> jnp.ndarray:
        """Total quantum potential energy (scalar) for autodiff."""
        t = theta_flat.reshape(K, n_h)
        diff = t[:, None, :] - t[None, :, :]  # (K, K, n_h)
        kernel = jnp.exp(kappa * jnp.cos(diff))  # (K, K, n_h)
        rho = jnp.mean(kernel, axis=1) + 1e-8  # (K, n_h)
        sqrt_rho = jnp.sqrt(rho)

        d2_kernel = kappa * (kappa * jnp.sin(diff) ** 2
                             - jnp.cos(diff)) * kernel
        d1_kernel = -kappa * jnp.sin(diff) * kernel

        grad_rho = jnp.mean(d1_kernel, axis=1)
        lapl_rho = jnp.mean(d2_kernel, axis=1)

        lapl_sqrt_rho = (lapl_rho / (2.0 * sqrt_rho)
                         - grad_rho ** 2 / (4.0 * rho * sqrt_rho))

        Q = -lapl_sqrt_rho / sqrt_rho
        return jnp.sum(Q)

    F_Q = -jax.grad(_Q_total)(theta.flatten())
    return Q_scale * F_Q.reshape(K, n_h)


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

    # Quantum potential: non-local anti-bunching force from Bohmian Q
    Q = quantum_potential(state.theta)  # (K, n_hidden)
    # Q strength is naturally proportional to density peakedness
    # (no artificial scaling needed with real Bohmian Q)

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


def cluster_coherence_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Complex phase-coherence phasor for cluster mean phases.

    Returns exp(i(psi_k - psi_l)) WITHOUT the modulus, so the caller can
    time-average the complex values and take |.| afterward. Phase-locked
    pairs survive averaging (|mean| -> 1); drifting pairs cancel (|mean| -> 0).

    Args:
        theta: (K, n_hidden) oscillator phases

    Returns:
        (K, K) complex phasor matrix.
    """
    psi = jnp.angle(jnp.mean(jnp.exp(1j * theta), axis=1))  # (K,)
    diff = psi[:, None] - psi[None, :]  # (K, K)
    return jnp.exp(1j * diff)  # complex — caller takes |.| after time-averaging


def unity_index(C: jnp.ndarray) -> tuple[float, float]:
    """Unity index and eigenvalue gap from coherence matrix.

    U = lambda_1 / sum(lambda_k)  — dominance of leading mode.
    gap = (lambda_1 - lambda_2) / lambda_1  — separation.

    U -> 1 with large gap = unified cognitive state (one subject).
    Multiple comparable eigenvalues = fragmented.

    Args:
        C: (K, K) symmetric coherence matrix

    Returns:
        (U, gap) both in [0, 1]
    """
    eigenvalues = jnp.linalg.eigvalsh(C)
    eigenvalues = jnp.flip(eigenvalues)  # descending
    total = jnp.sum(eigenvalues)
    U = eigenvalues[0] / (total + 1e-8)
    gap = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-8)
    return float(U), float(gap)
