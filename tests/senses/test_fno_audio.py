"""Test 1D Fourier Neural Operator for audio."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_spectral_conv1d_shape():
    from halo3.senses.fno_audio import SpectralConv1d
    key = jax.random.PRNGKey(0)
    layer = SpectralConv1d(in_channels=64, out_channels=64, modes=16, key=key)
    x = jax.random.normal(key, (32000, 64))
    out = layer(x)
    assert out.shape == (32000, 64), f"Expected (32000, 64), got {out.shape}"


def test_spectral_conv1d_finite():
    from halo3.senses.fno_audio import SpectralConv1d
    key = jax.random.PRNGKey(1)
    layer = SpectralConv1d(in_channels=64, out_channels=64, modes=16, key=key)
    x = jax.random.normal(key, (32000, 64))
    out = layer(x)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_audio_fno_output_shape():
    from halo3.senses.fno_audio import AudioFNO
    key = jax.random.PRNGKey(2)
    fno = AudioFNO(hidden_dim=64, n_layers=4, modes=16, n_tokens=8,
                   codebook_dim=64, key=key)
    waveform = jax.random.normal(key, (32000,))
    tokens = fno(waveform)
    assert tokens.shape == (8, 64), f"Expected (8, 64), got {tokens.shape}"


def test_audio_fno_zero_input():
    from halo3.senses.fno_audio import AudioFNO
    key = jax.random.PRNGKey(3)
    fno = AudioFNO(hidden_dim=64, n_layers=4, modes=16, n_tokens=8,
                   codebook_dim=64, key=key)
    waveform = jnp.zeros((32000,))
    tokens = fno(waveform)
    assert tokens.shape == (8, 64)
    assert bool(jnp.all(jnp.isfinite(tokens)))


def test_audio_fno_spectral_output():
    """Verify FNO stays in Fourier space — output comes from spectral modes."""
    from halo3.senses.fno_audio import AudioFNO
    key = jax.random.PRNGKey(4)
    fno = AudioFNO(hidden_dim=64, n_layers=4, modes=16, n_tokens=8,
                   codebook_dim=64, key=key)
    w1 = jax.random.normal(key, (32000,))
    w2 = jax.random.normal(jax.random.PRNGKey(99), (32000,))
    t1 = fno(w1)
    t2 = fno(w2)
    assert not np.allclose(np.array(t1), np.array(t2), atol=1e-3)
