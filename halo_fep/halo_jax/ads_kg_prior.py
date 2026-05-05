# halo_fep/halo_jax/ads_kg_prior.py
"""AdS Klein-Gordon flow prior — pure JAX function, no parameters."""
import jax
import jax.numpy as jnp

_EPS = 1e-6


def ads_kg_prior(
    x_noise: jnp.ndarray,   # (N_tok, d_boundary)
    x_data: jnp.ndarray,    # (N_tok, d_boundary)
    t: float,               # flow time in [0, 1]
    delta_flow: float,      # conformal dimension of the KG prior
) -> jnp.ndarray:           # (N_tok, d_boundary)
    """Compute AdS-KG prior flow vector field."""
    z_t     = 1.0 - t
    target  = x_data - x_noise                                  # (N_tok, d_boundary)
    dist2   = jnp.sum((x_data - x_noise) ** 2, axis=-1)        # (N_tok,)
    K       = (z_t / (z_t ** 2 + dist2 + _EPS)) ** delta_flow  # (N_tok,)
    weights = jax.nn.softmax(K)                                 # (N_tok,)
    v_kg    = weights[:, None] * target                         # (N_tok, d_boundary)
    return v_kg
