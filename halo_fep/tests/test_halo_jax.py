import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig
from halo_fep.halo_jax.holo_embedding import HoloEmbedding

cfg = HaloFEPConfig()
key = jax.random.PRNGKey(0)

def test_holo_embedding_shapes():
    model = HoloEmbedding(cfg, key)
    h = jnp.ones((cfg.n_tokens, cfg.d_model))
    x, z = model(h)
    assert x.shape == (cfg.n_tokens, cfg.d_boundary)
    assert z.shape == (cfg.n_tokens, 1)

def test_holo_embedding_z_in_range():
    model = HoloEmbedding(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    _, z = model(h)
    assert jnp.all(z > 0.0) and jnp.all(z < 1.0)

def test_holo_embedding_no_nan():
    model = HoloEmbedding(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x, z = model(h)
    assert not jnp.any(jnp.isnan(x))
    assert not jnp.any(jnp.isnan(z))

from halo_fep.halo_jax.holo_attention import HoloAttention

def test_holo_attention_shape():
    model = HoloAttention(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    out = model(h, x, z)
    assert out.shape == (cfg.n_tokens, cfg.d_model)

def test_holo_attention_no_nan():
    model = HoloAttention(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    out = model(h, x, z)
    assert not jnp.any(jnp.isnan(out))

def test_holo_attention_action_bias_changes_output():
    model = HoloAttention(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    delta_x = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_boundary))
    out_no_bias = model(h, x, z, delta_x=jnp.zeros_like(delta_x))
    out_biased  = model(h, x, z, delta_x=delta_x)
    assert not jnp.allclose(out_no_bias, out_biased)
