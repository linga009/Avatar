"""Tests for PageCurveMemory — ring buffer with participation-ratio eviction."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.page_memory import PageCurveMemory

_CFG = Halo3Config(d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8, d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4, n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4, mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2, meta_n_hidden=4, meta_n_actions=2, meta_k=3, max_cache=8, island_size=4)
_KEY = jax.random.PRNGKey(0)


def test_init_shapes():
    """init_state returns correct shapes for cache and island."""
    mem = PageCurveMemory(_CFG)
    state = mem.init_state()
    assert state.cache.shape == (_CFG.max_cache, _CFG.d_model)
    assert state.island.shape == (_CFG.island_size, _CFG.d_model)
    assert state.n_cached == 0
    assert state.island_ptr == 0


def test_write_increments():
    """Each write increments n_cached and places the vector."""
    mem = PageCurveMemory(_CFG)
    state = mem.init_state()
    x = jax.random.normal(_KEY, (_CFG.d_model,))
    state = mem(x, state)
    assert state.n_cached == 1
    assert jnp.allclose(state.cache[0], x)
    y = jax.random.normal(jax.random.PRNGKey(1), (_CFG.d_model,))
    state = mem(y, state)
    assert state.n_cached == 2
    assert jnp.allclose(state.cache[1], y)


def test_eviction_prefers_low_entropy():
    """A spike vector (energy in one dim) should be evicted before a spread vector.

    Participation ratio: PR = (sum x^2)^2 / (sum x^4).
    For a spike with all energy in 1 dim: PR = 1.
    For a spread vector (uniform): PR = d_model.
    s_gen = sq * pr, so spike gets a lower score and is evicted first.
    """
    mem = PageCurveMemory(_CFG)
    state = mem.init_state()

    # Fill cache: slot 0 gets a spike, slots 1..7 get spread vectors
    spike = jnp.zeros(_CFG.d_model).at[0].set(10.0)
    state = mem(spike, state)
    for i in range(1, _CFG.max_cache):
        spread = jnp.ones(_CFG.d_model) * 0.5
        state = mem(spread, state)

    # Cache is now full (n_cached=8). Spike is at slot 0.
    # Next write goes to slot 0 (8 % 8 = 0), overwriting the spike.
    # But we want the spike to survive for eviction scoring.
    # Instead, put the spike at slot 3 (won't be overwritten by next write at slot 0).
    state = state._replace(cache=state.cache.at[3].set(spike))

    # Write a new spread vector — triggers eviction. write_ptr = 8 % 8 = 0.
    new_vec = jnp.ones(_CFG.d_model) * 0.5
    state = mem(new_vec, state)

    # The spike at slot 3 has PR~1, sq=100, s_gen=100*1=100.
    # Spread vectors have sq=64*0.25=16, PR=64, s_gen=16*64=1024.
    # So spike has the lowest s_gen and should be evicted to island.
    evicted = state.island[0]
    assert jnp.max(evicted) > 5.0, "Spike should have been evicted to island"


def test_no_nan_after_overflow():
    """Writing many more vectors than max_cache does not produce NaNs."""
    mem = PageCurveMemory(_CFG)
    state = mem.init_state()
    for i in range(_CFG.max_cache * 5):
        x = jax.random.normal(jax.random.PRNGKey(i), (_CFG.d_model,))
        state = mem(x, state)
    assert not jnp.any(jnp.isnan(state.cache))
    assert not jnp.any(jnp.isnan(state.island))
    assert state.n_cached == _CFG.max_cache * 5


def test_zero_vector_safe():
    """Writing a zero vector does not produce NaN (0/0 in participation ratio)."""
    mem = PageCurveMemory(_CFG)
    state = mem.init_state()
    zero = jnp.zeros(_CFG.d_model)
    state = mem(zero, state)
    assert not jnp.any(jnp.isnan(state.cache))
    # Fill up and trigger eviction with zeros
    for _ in range(_CFG.max_cache + 2):
        state = mem(zero, state)
    assert not jnp.any(jnp.isnan(state.cache))
    assert not jnp.any(jnp.isnan(state.island))
