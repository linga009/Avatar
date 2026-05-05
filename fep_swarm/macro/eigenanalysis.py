import jax
import jax.numpy as jnp
from fep_swarm.agent.belief_update import free_energy
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.config import FEPConfig


def swarm_belief_rates(
    mu_flat: jnp.ndarray,      # [N * n_hidden]
    obs_indices: jnp.ndarray,  # [N]
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> jnp.ndarray:
    """mu_dot = -grad_mu F for all N agents, flattened to [N * n_hidden]."""
    mu = mu_flat.reshape(cfg.n_agents, cfg.n_hidden)
    grad_F = jax.grad(free_energy)
    mu_dot = jax.vmap(lambda m, o: -grad_F(m, o, gm))(mu, obs_indices)
    return mu_dot.reshape(-1)


def compute_jacobian(
    mu: jnp.ndarray,            # [N, n_hidden]
    obs_indices: jnp.ndarray,   # [N]
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> tuple:
    """
    J = d(mu_dot)/d(mu) at the current state.
    Returns (eigenvalues [N*d complex], gap_ratio scalar, magnitudes [N*d]).
    """
    mu_flat = mu.reshape(-1)
    f = lambda m: swarm_belief_rates(m, obs_indices, gm, cfg)
    J = jax.jacobian(f)(mu_flat)                 # [N*d, N*d]
    eigenvalues = jnp.linalg.eigvals(J)          # complex [N*d]
    magnitudes = jnp.abs(eigenvalues)
    gap = magnitudes.max() / (magnitudes.min() + 1e-8)
    return eigenvalues, gap, magnitudes


def temporal_horizons(
    magnitudes: jnp.ndarray,
    cfg: FEPConfig,
) -> tuple:
    """
    micro_horizon = 1 / max(|lambda|)   fastest mode — individual agent timescale
    macro_horizon = 1 / min(|lambda|)   slowest mode — global brain timescale
    """
    micro_h = 1.0 / (magnitudes.max() + 1e-8)
    macro_h = 1.0 / (jnp.where(magnitudes > 1e-6, magnitudes, jnp.inf).min() + 1e-8)
    return micro_h, macro_h
