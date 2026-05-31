"""Tests for ObsBridge with phase projection."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.bridge.obs_bridge import ObsBridge

_CFG = Halo3Config(
    d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
    d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
    n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
    mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
    meta_n_hidden=4, meta_n_actions=2, meta_k=3,
    max_cache=8, island_size=4,
)
_KEY = jax.random.PRNGKey(0)


def test_obs_bridge_shape():
    bridge = ObsBridge(_CFG, _KEY)
    h_out = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_model))
    obs = bridge(h_out)
    assert obs.shape == (_CFG.n_clusters, _CFG.n_obs)


def test_obs_bridge_phase_bounded():
    bridge = ObsBridge(_CFG, _KEY)
    h_out = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_model)) * 10.0
    obs = bridge(h_out)
    assert jnp.all(obs >= -jnp.pi - 1e-5)
    assert jnp.all(obs <= jnp.pi + 1e-5)


def test_obs_bridge_no_nan():
    bridge = ObsBridge(_CFG, _KEY)
    h_out = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_model))
    obs = bridge(h_out)
    assert jnp.all(jnp.isfinite(obs))


def test_obs_bridge_gradients_flow():
    import equinox as eqx
    bridge = ObsBridge(_CFG, _KEY)
    h_out = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_model))
    loss_fn = lambda b: jnp.sum(b(h_out) ** 2)
    grads = eqx.filter_grad(loss_fn)(bridge)
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0.0) for g in leaves if hasattr(g, 'shape') and g.shape != ())
