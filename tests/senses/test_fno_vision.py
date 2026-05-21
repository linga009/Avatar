"""Test 2D Fourier Neural Operator for vision."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_spectral_conv2d_shape():
    from halo3.senses.fno_vision import SpectralConv2d
    key = jax.random.PRNGKey(0)
    layer = SpectralConv2d(in_channels=64, out_channels=64, modes1=8, modes2=8, key=key)
    x = jax.random.normal(key, (224, 224, 64))
    out = layer(x)
    assert out.shape == (224, 224, 64), f"Expected (224, 224, 64), got {out.shape}"


def test_spectral_conv2d_finite():
    from halo3.senses.fno_vision import SpectralConv2d
    key = jax.random.PRNGKey(1)
    layer = SpectralConv2d(in_channels=64, out_channels=64, modes1=8, modes2=8, key=key)
    x = jax.random.normal(key, (224, 224, 64))
    out = layer(x)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_vision_fno_output_shape():
    from halo3.senses.fno_vision import VisionFNO
    key = jax.random.PRNGKey(2)
    fno = VisionFNO(hidden_dim=64, n_layers=4, modes=8, n_tokens=4,
                    codebook_dim=64, key=key)
    image = jax.random.normal(key, (224, 224, 3))
    tokens = fno(image)
    assert tokens.shape == (4, 64), f"Expected (4, 64), got {tokens.shape}"


def test_vision_fno_zero_input():
    from halo3.senses.fno_vision import VisionFNO
    key = jax.random.PRNGKey(3)
    fno = VisionFNO(hidden_dim=64, n_layers=4, modes=8, n_tokens=4,
                    codebook_dim=64, key=key)
    image = jnp.zeros((224, 224, 3))
    tokens = fno(image)
    assert tokens.shape == (4, 64)
    assert bool(jnp.all(jnp.isfinite(tokens)))


def test_vision_fno_different_images_different_tokens():
    from halo3.senses.fno_vision import VisionFNO
    key = jax.random.PRNGKey(4)
    fno = VisionFNO(hidden_dim=64, n_layers=4, modes=8, n_tokens=4,
                    codebook_dim=64, key=key)
    img1 = jax.random.normal(key, (224, 224, 3))
    img2 = jax.random.normal(jax.random.PRNGKey(99), (224, 224, 3))
    t1 = fno(img1)
    t2 = fno(img2)
    assert not np.allclose(np.array(t1), np.array(t2), atol=1e-3)
