# halo_fep/tests/test_model.py
import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig
from halo_fep.loss import halo_loss

cfg = HaloFEPConfig()
key = jax.random.PRNGKey(0)

def test_halo_loss_is_scalar():
    v_pred      = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target    = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w      = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    total, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert total.shape == ()

def test_halo_loss_no_nan():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    total, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert not jnp.isnan(total)
    for v in parts.values():
        assert not jnp.isnan(v)

def test_halo_loss_parts_keys():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    _, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert set(parts.keys()) == {"fm", "bek", "thermo", "page"}


import equinox as eqx
from halo_fep.model import HaloFEPModel, HaloFEPCarry, halo_fep_step

def _make_model_and_carry():
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    return model, carry

def test_model_init():
    model, carry = _make_model_and_carry()
    assert carry.swarm_mu.shape  == (cfg.n_agents, cfg.n_hidden)
    assert carry.swarm_action.shape == (cfg.n_agents, cfg.n_actions)

def test_closed_loop_step_shape():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    assert h_out.shape    == (cfg.n_tokens, cfg.d_model)
    assert obs.shape      == (cfg.n_agents,)
    assert v_pred.shape   == (cfg.n_tokens, cfg.d_model)
    assert v_target.shape == (cfg.n_tokens, cfg.d_model)

def test_closed_loop_step_no_nan():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    assert not jnp.any(jnp.isnan(h_out))
    assert not jnp.any(jnp.isnan(v_pred))

def test_closed_loop_jit_compiles():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    jit_step = eqx.filter_jit(halo_fep_step)
    new_carry, outputs = jit_step(model, carry, tokens, key)
    assert outputs[0].shape == (cfg.n_tokens, cfg.d_model)
