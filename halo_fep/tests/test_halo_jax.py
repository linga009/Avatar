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

from halo_fep.halo_jax.simple_ssm import SimpleSSM

def test_simple_ssm_shape():
    model = SimpleSSM(cfg, key)
    xs = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    ys = model(xs)
    assert ys.shape == (cfg.n_tokens, cfg.d_model)

def test_simple_ssm_no_nan():
    model = SimpleSSM(cfg, key)
    xs = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    ys = model(xs)
    assert not jnp.any(jnp.isnan(ys))

def test_simple_ssm_scan_matches_loop():
    """scan output must match manual Python loop."""
    model = SimpleSSM(cfg, key)
    xs = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    ys_scan = model(xs)
    h = jnp.zeros(cfg.d_state)
    ys_loop = []
    for i in range(cfg.n_tokens):
        h = jnp.exp(model.A) * h + model.B(xs[i])
        ys_loop.append(model.C(h) + model.D * xs[i])
    ys_loop = jnp.stack(ys_loop)
    assert jnp.allclose(ys_scan, ys_loop, atol=1e-5)

from halo_fep.halo_jax.ads_kg_prior import ads_kg_prior

def test_ads_kg_prior_shape():
    x_noise = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    x_data  = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_boundary))
    v_kg = ads_kg_prior(x_noise, x_data, t=0.5, delta_flow=cfg.delta_flow)
    assert v_kg.shape == (cfg.n_tokens, cfg.d_boundary)

def test_ads_kg_prior_no_nan():
    x_noise = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    x_data  = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_boundary))
    v_kg = ads_kg_prior(x_noise, x_data, t=0.5, delta_flow=cfg.delta_flow)
    assert not jnp.any(jnp.isnan(v_kg))

def test_ads_kg_prior_t0_no_nan():
    x_noise = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    x_data  = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_boundary))
    v_kg = ads_kg_prior(x_noise, x_data, t=0.0, delta_flow=cfg.delta_flow)
    assert v_kg.shape == (cfg.n_tokens, cfg.d_boundary)
    assert not jnp.any(jnp.isnan(v_kg))

from halo_fep.halo_jax.page_memory import PageCurveMemory, PageMemState

def test_page_memory_init_state():
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    assert state.cache.shape == (cfg.max_cache, cfg.d_model)
    assert state.island.shape == (cfg.island_size, cfg.d_model)
    assert int(state.n_cached) == 0

def test_page_memory_add_single_token():
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    x_i = jax.random.normal(key, (cfg.d_model,))
    state2 = mem(x_i, state)
    assert int(state2.n_cached) == 1
    assert jnp.allclose(state2.cache[0], x_i)

def test_page_memory_cache_never_exceeds_max():
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    for i in range(cfg.max_cache + 10):
        x_i = jax.random.normal(jax.random.PRNGKey(i), (cfg.d_model,))
        state = mem(x_i, state)
    valid_slots = jnp.sum(jnp.any(state.cache != 0.0, axis=-1))
    assert valid_slots <= cfg.max_cache

def test_page_memory_island_fills_on_eviction():
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    for i in range(cfg.max_cache + 5):
        x_i = jax.random.normal(jax.random.PRNGKey(i), (cfg.d_model,))
        state = mem(x_i, state)
    assert int(state.island_ptr) > 0

from halo_fep.halo_jax.backbone import HALOBackbone

def test_backbone_output_shape():
    model = HALOBackbone(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    h_out = model(h, x, z)
    assert h_out.shape == (cfg.n_tokens, cfg.d_model)

def test_backbone_no_nan():
    model = HALOBackbone(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    h_out = model(h, x, z)
    assert not jnp.any(jnp.isnan(h_out))

def test_backbone_jit_compiles():
    model = HALOBackbone(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    jit_fn = eqx.filter_jit(model)
    h_out = jit_fn(h, x, z)
    assert h_out.shape == (cfg.n_tokens, cfg.d_model)

def test_backbone_action_bias_changes_output():
    model = HALOBackbone(cfg, key)
    h = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))
    z = jnp.ones((cfg.n_tokens, 1)) * 0.5
    delta_x = jax.random.normal(jax.random.PRNGKey(99), (cfg.n_tokens, cfg.d_boundary))
    out_no_bias = model(h, x, z, delta_x=jnp.zeros_like(delta_x))
    out_biased  = model(h, x, z, delta_x=delta_x)
    assert not jnp.allclose(out_no_bias, out_biased)
