"""Hamiltonian Neural ODE — symplectic leapfrog integrator on AdS space."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from halo3.config import Halo3Config
from halo3.lorentz_ops import lorentz_distance


def v_ads(q: jnp.ndarray, kappa: float) -> jnp.ndarray:
    """AdS potential — sum of pairwise log-cosh geodesic distances.

    Parameters
    ----------
    q:     (N, d) position matrix on the hyperboloid
    kappa: curvature parameter

    Returns
    -------
    Scalar potential value.
    """
    # Vectorized pairwise distances — no Python for-loops over N
    def _dist_row(qi):
        return jax.vmap(lambda qj: lorentz_distance(qi, qj, kappa))(q)

    dist_matrix = jax.vmap(_dist_row)(q)          # (N, N)
    mask = jnp.triu(jnp.ones_like(dist_matrix), k=1)  # upper triangle only
    return jnp.sum(mask * jnp.log(jnp.cosh(dist_matrix) + 1e-6))


class LearnedHamiltonian(eqx.Module):
    """H(q,p) = KE + V_AdS(q) + V_learned(q).

    Parameters learned by two-layer MLP acting on mean-pooled positions.
    """

    v_fc1: eqx.nn.Linear   # d_boundary -> 64
    v_fc2: eqx.nn.Linear   # 64 -> 1
    kappa: float = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.v_fc1 = eqx.nn.Linear(cfg.d_boundary, 64, key=k1)
        self.v_fc2 = eqx.nn.Linear(64, 1, key=k2)
        self.kappa = cfg.init_curvature

    def _v_learned(self, q: jnp.ndarray) -> jnp.ndarray:
        q_mean = jnp.mean(q, axis=0)            # (d_boundary,)
        return self.v_fc2(jax.nn.silu(self.v_fc1(q_mean))).squeeze()

    def __call__(self, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
        T = 0.5 * jnp.sum(p ** 2)
        V = v_ads(q, self.kappa) + self._v_learned(q)
        return T + V


def leapfrog_step(
    H: LearnedHamiltonian,
    q: jnp.ndarray,
    p: jnp.ndarray,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single symplectic leapfrog (Störmer–Verlet) step."""
    # Gradient of V = H - KE  (KE uses current p, so subtract it)
    dVdq = jax.grad(lambda qq: H(qq, p) - 0.5 * jnp.sum(p ** 2))(q)
    p_half = p - (eps / 2) * dVdq

    q_new = q + eps * p_half

    dVdq_new = jax.grad(lambda qq: H(qq, p_half) - 0.5 * jnp.sum(p_half ** 2))(q_new)
    p_new = p_half - (eps / 2) * dVdq_new

    return q_new, p_new


def leapfrog_integrate(
    H: LearnedHamiltonian,
    q: jnp.ndarray,
    p: jnp.ndarray,
    n_steps: int,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate Hamilton's equations for n_steps steps using leapfrog.

    Uses jax.lax.scan for XLA-friendly, differentiable integration.
    """
    def step(carry, _):
        q, p = carry
        return leapfrog_step(H, q, p, eps), None

    (q_f, p_f), _ = jax.lax.scan(step, (q, p), None, length=n_steps)
    return q_f, p_f


class MomentumInitializer(eqx.Module):
    """Project backbone hidden states to momentum vectors in d_boundary space."""

    proj: eqx.nn.Linear   # d_model -> d_boundary

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        self.proj = eqx.nn.Linear(cfg.d_model, cfg.d_boundary, key=key)

    def __call__(self, h_out: jnp.ndarray) -> jnp.ndarray:
        """Map (N, d_model) hidden states to (N, d_boundary) momenta."""
        return jax.vmap(self.proj)(h_out)
