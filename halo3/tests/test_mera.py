"""Tests for MERA tensor train FFN."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.mera_ffn import MERAFFN

_CFG = Halo3Config(d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8, d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4, n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4, mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2, meta_n_hidden=4, meta_n_actions=2, meta_k=3, max_cache=8, island_size=4)
_KEY = jax.random.PRNGKey(0)


def test_output_shape():
    ffn = MERAFFN(_CFG, _KEY)
    x = jax.random.normal(_KEY, (_CFG.d_model,))
    assert ffn(x).shape == (_CFG.d_model,)


def test_no_nan():
    ffn = MERAFFN(_CFG, _KEY)
    x = jax.random.normal(_KEY, (_CFG.d_model,))
    assert jnp.all(jnp.isfinite(ffn(x)))


def test_fewer_params_than_dense():
    ffn = MERAFFN(_CFG, _KEY)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(ffn) if hasattr(p, 'size'))
    dense_params = 3 * _CFG.d_model * _CFG.d_model
    assert n_params < dense_params / 2


def test_gradients_flow():
    ffn = MERAFFN(_CFG, _KEY)
    x = jax.random.normal(_KEY, (_CFG.d_model,))
    grads = jax.grad(lambda f: jnp.sum(f(x)))(ffn)
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0.0) for g in leaves if hasattr(g, 'shape'))


def test_different_inputs_different_outputs():
    ffn = MERAFFN(_CFG, _KEY)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (_CFG.d_model,))
    x2 = jax.random.normal(jax.random.PRNGKey(2), (_CFG.d_model,))
    assert not jnp.allclose(ffn(x1), ffn(x2))
