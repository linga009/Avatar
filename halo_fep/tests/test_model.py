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
