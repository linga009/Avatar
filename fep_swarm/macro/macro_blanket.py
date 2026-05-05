import jax
import jax.numpy as jnp
from fep_swarm.macro.renormalization import MacroState
from fep_swarm.agent.belief_update import free_energy
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.config import FEPConfig


def macro_free_energy(
    macro: MacroState,
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> jnp.ndarray:
    """F_macro: free energy of the macro agent (mean over groups)."""
    n_groups = macro.M.shape[0]

    def group_F(g):
        mu_g = macro.M[g]
        soft_obs = jax.nn.softmax(macro.S[g])
        return free_energy(mu_g, soft_obs, gm)

    F_groups = jax.vmap(group_F)(jnp.arange(n_groups))
    return F_groups.sum()


def micro_free_energy_sum(
    mu: jnp.ndarray,         # [N, n_hidden]
    soft_obs: jnp.ndarray,   # [N, n_obs] float32
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> jnp.ndarray:
    """Sum of F_i over all N agents."""
    F_all = jax.vmap(lambda m, o: free_energy(m, o, gm))(mu, soft_obs)
    return F_all.sum()


def check_macro_bound(
    F_macro: jnp.ndarray,
    F_micro_sum: jnp.ndarray,
    I_sync: jnp.ndarray,
) -> tuple:
    """
    Verify: F_macro <= sum(F_i) - I(synchrony)
    Returns (holds: bool scalar, violation: float scalar).
    violation > 0 means the bound is violated.
    """
    rhs = F_micro_sum - I_sync
    violation = F_macro - rhs
    return violation <= 0.0, violation
