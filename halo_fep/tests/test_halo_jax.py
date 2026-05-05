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
