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
