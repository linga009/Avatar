# halo_fep/loss.py
"""HALO loss components ported to JAX.

L_total = L_FM + lambda_bek * L_Bek + lambda_thermo * L_thermo + lambda_page * L_page

L_FM     = MSE(v_pred, v_target)
L_Bek    = mean(max(0, H(A^l) - alpha*N*d_head/n_heads))  Bekenstein regularizer
L_thermo = max(0, 0.01 - eps_prod)          entropy production lower bound
L_page   = KL(evict_scores || s_gen_dist)   page curve alignment
"""
import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig


def halo_loss(
    v_pred: jnp.ndarray,        # (N_tok, d_model) predicted flow field
    v_target: jnp.ndarray,      # (N_tok, d_model) target flow field
    attn_weights: jnp.ndarray,  # (N_tok, N_tok) attention weight matrix
    evict_scores: jnp.ndarray,  # (N_tok,) eviction score distribution
    cfg: HaloFEPConfig,
) -> tuple:
    """Returns (total_loss, {"fm", "bek", "thermo", "page"})."""
    # Flow matching
    L_fm = jnp.mean((v_pred - v_target) ** 2)

    # Bekenstein: attention entropy <= alpha * N * d_head / n_heads
    H_attn = -jnp.sum(
        attn_weights * jnp.log(attn_weights + 1e-8), axis=-1
    )  # (N_tok,)
    bek_bound = cfg.bekenstein_alpha * attn_weights.shape[0] * cfg.d_head / cfg.n_heads
    L_bek = jnp.mean(jnp.maximum(0.0, H_attn - bek_bound))

    # Entropy production lower bound
    eps_prod = jnp.mean(jnp.sum(v_pred ** 2, axis=-1) / 2.0)
    L_thermo = jnp.maximum(0.0, 0.01 - eps_prod)

    # Page curve alignment: KL(evict_scores || s_gen_dist)
    s_gen     = jnp.sum(v_pred ** 2, axis=-1) * cfg.d_head / 4.0  # (N_tok,) proxy
    s_gen_dist = jax.nn.softmax(-s_gen)                            # evict low-S tokens
    L_page = jnp.sum(
        evict_scores * (jnp.log(evict_scores + 1e-8) - jnp.log(s_gen_dist + 1e-8))
    )

    total = L_fm + cfg.lambda_bek * L_bek + cfg.lambda_thermo * L_thermo + cfg.lambda_page * L_page
    return total, {"fm": L_fm, "bek": L_bek, "thermo": L_thermo, "page": L_page}


def unified_elbo_loss(
    model,
    carry,
    tokens: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple:
    """Unified ELBO: L_ELBO = L_flow + L_obs + L_prior.

    L_flow  = mean ||v_pred - v_target||^2           (flow matching)
    L_obs   = -mean einsum('ao,oi,ai->a',             (HALO->FEP bridge)
                           soft_obs, log_A, q_eta)
    L_prior = mean KL[q(eta) || D]                   (FEP prior)

    Replaces the ad-hoc L_HALO + lambda_fep * F_swarm.
    Returns (total_loss, {"l_flow": ..., "l_obs": ..., "l_prior": ...}).
    """
    from halo_fep.model import halo_fep_step  # local to avoid circular import at module level

    new_carry, (h_out, soft_obs, v_pred, v_target) = halo_fep_step(
        model, carry, tokens, key
    )

    # Flow matching term
    l_flow = jnp.mean((v_pred - v_target) ** 2)

    # Soft observation likelihood — bridges HALO output to FEP beliefs
    # soft_obs: (N_agents, n_obs), log_A: (n_obs, n_hidden), q_eta: (N_agents, n_hidden)
    q_eta = jax.nn.softmax(new_carry.swarm_mu)            # (N_agents, n_hidden)
    log_A = jnp.log(model.gm.A + 1e-8)                   # (n_obs, n_hidden)
    l_obs = -jnp.mean(
        jnp.einsum('ao,oi,ai->a', soft_obs, log_A, q_eta)
    )

    # KL prior term
    log_q = jnp.log(q_eta + 1e-8)                         # (N_agents, n_hidden)
    log_D = jnp.log(model.gm.D + 1e-8)  # (n_hidden,)
    l_prior = jnp.mean(jnp.sum(q_eta * (log_q - log_D), axis=-1))

    total = l_flow + l_obs + l_prior
    return total, {"l_flow": l_flow, "l_obs": l_obs, "l_prior": l_prior}
