# halo_fep/tests/test_bridge.py
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig
from halo_fep.bridge.obs_bridge import ObsBridge

cfg = HaloFEPConfig()
key = jax.random.PRNGKey(42)

def test_obs_bridge_output_shape():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    s_i = bridge(h_out)
    assert s_i.shape == (cfg.n_agents,)

def test_obs_bridge_obs_in_range():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    s_i = bridge(h_out)
    assert jnp.all(s_i >= 0) and jnp.all(s_i < cfg.n_obs)

def test_obs_bridge_no_nan():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    s_i = bridge(h_out)
    assert not jnp.any(jnp.isnan(s_i.astype(jnp.float32)))

def test_obs_bridge_gradients_flow():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    # Gradient of sum(logits) wrt bridge params
    def loss_fn(bridge):
        logits = bridge._logits(h_out)   # raw logits before argmax
        return jnp.sum(logits)
    grads = eqx.filter_grad(loss_fn)(bridge)
    grad_vals = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_vals)
