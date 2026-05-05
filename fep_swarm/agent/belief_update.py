import jax
import jax.numpy as jnp
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.config import FEPConfig


def free_energy(
    mu: jnp.ndarray,
    obs_idx: int,
    gm: DiscreteGenerativeModel,
) -> jnp.ndarray:
    """
    F(μ, s) = KL[Q(η;μ) || P(η)] − E_Q[ln P(s|η)]
    mu: [n_hidden] log-unnormalized beliefs
    obs_idx: integer observation index
    """
    q_eta = jax.nn.softmax(mu)                          # Q(η;μ): [n_hidden]
    p_eta = gm.D                                         # P(η):   [n_hidden]
    kl = jnp.sum(q_eta * (jnp.log(q_eta + 1e-8) - jnp.log(p_eta + 1e-8)))

    # E_Q[ln P(s|η)] = Σ_η Q(η) · ln A[s, η]
    log_A_s = jnp.log(gm.A[obs_idx] + 1e-8)            # [n_hidden]
    expected_log_lik = jnp.sum(q_eta * log_A_s)

    return kl - expected_log_lik


def belief_update(
    mu_init: jnp.ndarray,
    obs_idx: int,
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> jnp.ndarray:
    """
    Gradient descent on F: μ_{t+1} = μ_t − lr · ∇_μ F(μ_t, s)
    Uses jax.lax.fori_loop for JIT compatibility.
    """
    grad_F = jax.grad(free_energy)

    def step(i, mu):
        return mu - cfg.inf_lr * grad_F(mu, obs_idx, gm)

    return jax.lax.fori_loop(0, cfg.inf_steps, step, mu_init)
