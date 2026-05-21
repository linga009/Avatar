"""Test contrastive alignment for speech-text binding."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_contrastive_loss_shape():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    audio_emb = jax.random.normal(jax.random.PRNGKey(0), (64,))
    text_emb = jax.random.normal(jax.random.PRNGKey(1), (64,))
    for i in range(5):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (64,)))
    loss = aligner.compute_loss(audio_emb, text_emb)
    assert loss.shape == ()
    assert np.isfinite(float(loss))


def test_contrastive_loss_positive():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    for i in range(8):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (64,)))
    audio = jax.random.normal(jax.random.PRNGKey(0), (64,))
    text = jax.random.normal(jax.random.PRNGKey(1), (64,))
    loss = aligner.compute_loss(audio, text)
    assert float(loss) > 0.0


def test_similar_pair_lower_loss():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    for i in range(8):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (64,)))
    audio = jax.random.normal(jax.random.PRNGKey(0), (64,))
    text_similar = audio + jax.random.normal(jax.random.PRNGKey(1), (64,)) * 0.1
    text_random = jax.random.normal(jax.random.PRNGKey(2), (64,))
    loss_similar = float(aligner.compute_loss(audio, text_similar))
    loss_random = float(aligner.compute_loss(audio, text_random))
    assert loss_similar < loss_random


def test_ring_buffer_capacity():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=4, tau=0.07)
    for i in range(10):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(i), (64,)))
    assert aligner.buffer_count == 4


def test_codebook_utilization():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    indices = jnp.array([i % 80 for i in range(160)])
    for chunk in indices.reshape(-1, 16):
        aligner.push_indices(chunk)
    util = aligner.compute_utilization(codebook_size=128)
    assert 0.5 < util < 0.7


def test_empty_buffer_returns_zero_loss():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    audio = jax.random.normal(jax.random.PRNGKey(0), (64,))
    text = jax.random.normal(jax.random.PRNGKey(1), (64,))
    loss = aligner.compute_loss(audio, text)
    assert float(loss) == 0.0
