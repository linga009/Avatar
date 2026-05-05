import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig

EPS = 1e-6


class HoloAttention(eqx.Module):
    """Multi-head attention using the bulk-to-boundary propagator K_Delta.

    A_ij = softmax_j( (z_i / (z_i^2 + ||x_i - x_j||^2 + eps))^delta )

    Conformal dimension delta is per-head and learnable (stored as log_delta).
    Action bias delta_x shifts boundary positions before kernel computation.
    """

    log_delta: jnp.ndarray   # (n_heads,)
    v_proj: eqx.nn.Linear    # d_model -> n_heads * d_head
    out_proj: eqx.nn.Linear  # n_heads * d_head -> d_model
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.log_delta = jnp.zeros(cfg.n_heads)
        self.v_proj   = eqx.nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, key=k1)
        self.out_proj = eqx.nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, key=k2)
        self.n_heads  = cfg.n_heads
        self.d_head   = cfg.d_head

    def __call__(
        self,
        h: jnp.ndarray,                        # (N_tok, d_model)
        x: jnp.ndarray,                        # (N_tok, d_boundary)
        z: jnp.ndarray,                        # (N_tok, 1)
        delta_x: jnp.ndarray | None = None,    # (N_tok, d_boundary) action bias
    ) -> jnp.ndarray:                           # (N_tok, d_model)
        N_tok = h.shape[0]
        x_b = x + delta_x if delta_x is not None else x  # (N_tok, d_boundary)
        z_i = z[:, 0]  # (N_tok,)

        # Value projections: (N_tok, n_heads * d_head) -> (n_heads, N_tok, d_head)
        V = jax.vmap(self.v_proj)(h)
        V_heads = V.reshape(N_tok, self.n_heads, self.d_head).transpose(1, 0, 2)

        # K_Delta base: (N_tok, N_tok) -- shared geometry, different delta per head
        diff  = x_b[:, None, :] - x_b[None, :, :]       # (N_tok, N_tok, d_boundary)
        dist2 = jnp.sum(diff ** 2, axis=-1)               # (N_tok, N_tok)
        base  = z_i[:, None] / (z_i[:, None] ** 2 + dist2 + EPS)  # (N_tok, N_tok)

        def head_attn(log_d: jnp.ndarray, v_h: jnp.ndarray) -> jnp.ndarray:
            # log_d: scalar, v_h: (N_tok, d_head)
            K = base ** jnp.exp(log_d)                   # (N_tok, N_tok)
            A = jax.nn.softmax(K, axis=-1)
            return A @ v_h                               # (N_tok, d_head)

        # vmap over heads: log_delta (n_heads,), V_heads (n_heads, N_tok, d_head)
        head_outs = jax.vmap(head_attn)(self.log_delta, V_heads)  # (n_heads, N_tok, d_head)
        concat = head_outs.transpose(1, 0, 2).reshape(N_tok, self.n_heads * self.d_head)
        return jax.vmap(self.out_proj)(concat)           # (N_tok, d_model)
