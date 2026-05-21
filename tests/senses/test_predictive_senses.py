"""Test that learn_from_error trains both model and sense_module."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from halo3.config import Halo3Config


def _small_cfg():
    return Halo3Config(
        d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
        d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
        n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
        mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
        meta_n_hidden=4, meta_n_actions=2, meta_k=3,
        max_cache=8, island_size=4,
        fno_hidden_dim=16, fno_n_layers=2, fno_audio_modes=4,
        fno_vision_modes=4, codebook_size_audio=16, codebook_size_vision=8, codebook_dim=16,
        n_audio_tokens=2, n_vision_tokens=2,
    )


def test_learn_from_error_returns_updated_sense_module():
    from halo3.model import Halo3Model
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule

    cfg = _small_cfg()
    key = jax.random.PRNGKey(0)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)

    predictor = PredictiveProcessor(lr=1e-5)

    text_tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
    audio_raw = jnp.zeros((32000,))
    vision_raw = jnp.zeros((224, 224, 3))
    q_actual = jnp.zeros((cfg.n_tokens, cfg.d_boundary))

    new_model, new_sm, loss, info = predictor.learn_from_error(
        model, sense_module, carry, text_tokens, audio_raw, vision_raw, q_actual, key)

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert new_sm is not None
    assert "audio_indices" in info


def test_learn_from_error_loss_is_finite():
    from halo3.model import Halo3Model
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule

    cfg = _small_cfg()
    key = jax.random.PRNGKey(1)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)

    predictor = PredictiveProcessor(lr=1e-5)
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_raw = jax.random.normal(key, (32000,))
    vision_raw = jax.random.normal(key, (224, 224, 3))
    q_actual = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))

    _, _, loss, _ = predictor.learn_from_error(
        model, sense_module, carry, text_tokens, audio_raw, vision_raw, q_actual, key)

    assert np.isfinite(loss), f"Loss was not finite: {loss}"


def test_learn_from_error_with_contrastive():
    from halo3.model import Halo3Model
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule
    from halo3.senses.contrastive_aligner import ContrastiveAligner

    cfg = _small_cfg()
    key = jax.random.PRNGKey(2)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)
    aligner = ContrastiveAligner(embed_dim=cfg.d_model, buffer_size=16, tau=0.07)
    for i in range(4):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (cfg.d_model,)))

    predictor = PredictiveProcessor(lr=1e-5)
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_raw = jax.random.normal(key, (32000,))
    vision_raw = jax.random.normal(key, (224, 224, 3))
    q_actual = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))

    new_model, new_sm, loss, info = predictor.learn_from_error(
        model, sense_module, carry, text_tokens, audio_raw, vision_raw, q_actual, key,
        contrastive_aligner=aligner, text_paired=True, contrastive_weight=0.3)
    assert np.isfinite(loss)
    assert info["text_paired"] is True
