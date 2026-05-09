"""GaLore — Gradient Low-Rank Projection for memory-efficient training."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx


def galore_projection(G: jnp.ndarray, rank: int) -> jnp.ndarray:
    """Compute projection matrix P from gradient G via SVD.
    Returns P of shape (m, rank) with orthonormal columns.
    """
    U, S, Vt = jnp.linalg.svd(G, full_matrices=False)
    return U[:, :rank]


def make_galore_mask(model) -> any:
    """Return pytree: True for GaLore-eligible weights (SwiGLU 2D matrices)."""
    def _is_ffn_weight(path, leaf):
        path_str = jtu.keystr(path)
        is_ffn = "ffns" in path_str and "weight" in path_str
        is_2d = hasattr(leaf, 'ndim') and leaf.ndim == 2
        return is_ffn and is_2d
    return jtu.tree_map_with_path(
        lambda path, leaf: _is_ffn_weight(path, leaf),
        eqx.filter(model, eqx.is_array),
    )


def apply_galore(grads, model, rank: int) -> any:
    """Apply GaLore projection to eligible gradient leaves."""
    mask = make_galore_mask(model)
    def _project(grad, is_galore):
        if not is_galore or not hasattr(grad, 'ndim') or grad.ndim != 2:
            return grad
        r = min(rank, min(grad.shape))
        P = galore_projection(grad, r)
        return P @ (P.T @ grad)
    return jtu.tree_map(_project, grads, mask)
