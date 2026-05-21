"""SenseProjections unit tests."""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest


def make_proj(key=None):
    from halo3.senses.projections import SenseProjections
    if key is None:
        key = jax.random.PRNGKey(0)
    return SenseProjections(audio_dim=768, vision_dim=768, d_model=2048, key=key)


def test_inject_shape():
    proj = make_proj()
    text = jnp.zeros((32, 2048))
    audio = jnp.zeros((8, 768))
    vision = jnp.zeros((768,))
    out = proj.inject(text, audio, vision)
    assert out.shape == (32, 2048), f"Expected (32, 2048), got {out.shape}"


def test_inject_zero_input_produces_zero_residual():
    """Zero audio+vision -> zero injection residual (use_bias=False on projs)."""
    proj = make_proj()
    text = jnp.ones((32, 2048))
    audio = jnp.zeros((8, 768))
    vision = jnp.zeros((768,))
    out = proj.inject(text, audio, vision)
    # Residual should be zero: sense_context = 0 -> gate * 0 = 0
    np.testing.assert_allclose(np.array(out), np.ones((32, 2048)), atol=1e-5)


def test_inject_finite():
    key = jax.random.PRNGKey(42)
    proj = make_proj(key)
    text = jax.random.normal(key, (32, 2048))
    audio = jax.random.normal(key, (8, 768))
    vision = jax.random.normal(key, (768,))
    out = proj.inject(text, audio, vision)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_save_load_roundtrip(tmp_path):
    from halo3.senses.projections import SenseProjections, save_sense_proj, load_sense_proj
    proj = make_proj()
    path = str(tmp_path / "sense_proj")
    save_sense_proj(proj, path)
    loaded = load_sense_proj(768, 512, 2048, path)
    # Check that weights match
    orig_leaves = jax.tree_util.tree_leaves(eqx.filter(proj, eqx.is_array))
    load_leaves = jax.tree_util.tree_leaves(eqx.filter(loaded, eqx.is_array))
    for o, l in zip(orig_leaves, load_leaves):
        np.testing.assert_allclose(np.array(o), np.array(l), atol=1e-6)
