"""Test spectral VQ-VAE codebook and quantization."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_codebook_init_shape():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(0)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    assert cb.embeddings.shape == (32, 64)


def test_quantize_output_shapes():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(0)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    z_e = jax.random.normal(key, (8, 64))
    z_q, indices, commitment_loss = cb.quantize(z_e)
    assert z_q.shape == (8, 64), f"Expected (8, 64), got {z_q.shape}"
    assert indices.shape == (8,), f"Expected (8,), got {indices.shape}"
    assert commitment_loss.shape == (), f"Expected scalar, got {commitment_loss.shape}"


def test_quantize_indices_in_range():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(1)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    z_e = jax.random.normal(key, (8, 64))
    _, indices, _ = cb.quantize(z_e)
    assert bool(jnp.all(indices >= 0))
    assert bool(jnp.all(indices < 32))


def test_quantize_straight_through():
    """z_q should have same value as looked-up embedding but grad flows to z_e."""
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(2)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    z_e = jax.random.normal(key, (8, 64))
    z_q, indices, _ = cb.quantize(z_e)
    expected = cb.embeddings[indices]
    np.testing.assert_allclose(np.array(z_q), np.array(expected), atol=1e-5)


def test_ema_update_changes_embeddings():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(3)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    old_emb = np.array(cb.embeddings)
    z_e = jax.random.normal(key, (8, 64)) * 10
    _, indices, _ = cb.quantize(z_e)
    cb_new = cb.ema_update(z_e, indices, decay=0.99)
    new_emb = np.array(cb_new.embeddings)
    assert not np.allclose(old_emb, new_emb, atol=1e-6)


def test_dead_code_revival():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(4)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    usage = jnp.zeros(32).at[:8].set(50.0)
    z_e = jax.random.normal(key, (8, 64))
    cb_new = cb.revive_dead_codes(usage, z_e, threshold=10, key=key)
    old_dead = np.array(cb.embeddings[8:])
    new_dead = np.array(cb_new.embeddings[8:])
    assert not np.allclose(old_dead, new_dead, atol=1e-6)


def test_commitment_loss_positive():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(5)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    z_e = jax.random.normal(key, (8, 64))
    _, _, commitment_loss = cb.quantize(z_e)
    assert float(commitment_loss) > 0.0


def test_zero_input_quantizes():
    from halo3.senses.spectral_vqvae import SpectralCodebook
    key = jax.random.PRNGKey(6)
    cb = SpectralCodebook(codebook_size=32, codebook_dim=64, key=key)
    z_e = jnp.zeros((4, 64))
    z_q, indices, loss = cb.quantize(z_e)
    assert z_q.shape == (4, 64)
    assert bool(jnp.all(jnp.isfinite(z_q)))
