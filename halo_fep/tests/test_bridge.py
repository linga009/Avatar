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
    soft_obs = bridge(h_out)
    assert soft_obs.shape == (cfg.n_agents, cfg.n_obs)

def test_obs_bridge_rows_sum_to_one():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    soft_obs = bridge(h_out)
    row_sums = jnp.sum(soft_obs, axis=-1)  # (N_agents,)
    assert jnp.allclose(row_sums, jnp.ones(cfg.n_agents), atol=1e-5)

def test_obs_bridge_no_nan():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    logits = bridge._logits(h_out)
    assert not jnp.any(jnp.isnan(logits))

def test_obs_bridge_gradients_flow():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    # Gradient of sum(logits) wrt bridge params
    def loss_fn(bridge):
        logits = bridge._logits(h_out)   # raw logits before softmax
        return jnp.sum(logits)
    grads = eqx.filter_grad(loss_fn)(bridge)
    grad_vals = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_vals)

from halo_fep.bridge.action_bridge import ActionBridge
from halo_fep.bridge.belief_bridge import BeliefBridge

def test_action_bridge_output_shape():
    bridge = ActionBridge(cfg, key)
    a_i = jax.random.normal(key, (cfg.n_agents, cfg.n_actions))
    delta_x = bridge(a_i)
    assert delta_x.shape == (cfg.n_tokens, cfg.d_boundary)

def test_action_bridge_no_nan():
    bridge = ActionBridge(cfg, key)
    a_i = jnp.ones((cfg.n_agents, cfg.n_actions)) / cfg.n_actions
    delta_x = bridge(a_i)
    assert not jnp.any(jnp.isnan(delta_x))

def test_action_bridge_gradients_flow():
    bridge = ActionBridge(cfg, key)
    a_i = jax.random.normal(key, (cfg.n_agents, cfg.n_actions))
    def loss_fn(bridge):
        return jnp.sum(bridge(a_i))
    grads = eqx.filter_grad(loss_fn)(bridge)
    grad_vals = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_vals)

def test_belief_bridge_output_shape():
    bridge = BeliefBridge(cfg, key)
    mu_i = jax.random.normal(key, (cfg.n_agents, cfg.n_hidden))
    delta_v = bridge(mu_i)
    assert delta_v.shape == (cfg.n_tokens, cfg.d_model)

def test_belief_bridge_no_nan():
    bridge = BeliefBridge(cfg, key)
    mu_i = jax.random.normal(key, (cfg.n_agents, cfg.n_hidden))
    delta_v = bridge(mu_i)
    assert not jnp.any(jnp.isnan(delta_v))

def test_belief_bridge_gradients_flow():
    bridge = BeliefBridge(cfg, key)
    mu_i = jax.random.normal(key, (cfg.n_agents, cfg.n_hidden))
    def loss_fn(bridge):
        return jnp.sum(bridge(mu_i))
    grads = eqx.filter_grad(loss_fn)(bridge)
    grad_vals = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert any(jnp.any(g != 0.0) for g in grad_vals)
