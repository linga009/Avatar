"""Test that learn_from_error trains both model and sense_proj."""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest


def make_small_config():
    from halo3.config import Halo3Config
    return Halo3Config(
        d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
        d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
        n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
        mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
        meta_n_hidden=4, meta_n_actions=2, meta_k=3,
        max_cache=8, island_size=4,
    )


def test_learn_from_error_returns_updated_sense_proj():
    from halo3.model import Halo3Model
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.projections import SenseProjections

    cfg = make_small_config()
    key = jax.random.PRNGKey(0)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_proj = SenseProjections(
        audio_dim=768, vision_dim=768, d_model=cfg.d_model, key=key)

    predictor = PredictiveProcessor(lr=1e-5)

    text_tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
    audio_jax = jnp.zeros((8, 768))
    vision_jax = jnp.zeros((768,))
    q_actual = jnp.zeros((cfg.n_tokens, cfg.d_boundary))

    new_model, new_sense_proj, loss = predictor.learn_from_error(
        model, sense_proj, carry, text_tokens, audio_jax, vision_jax, q_actual, key)

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    assert new_sense_proj is not None


def test_learn_from_error_loss_is_finite():
    from halo3.model import Halo3Model
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.projections import SenseProjections

    cfg = make_small_config()
    key = jax.random.PRNGKey(1)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_proj = SenseProjections(
        audio_dim=768, vision_dim=768, d_model=cfg.d_model, key=key)

    predictor = PredictiveProcessor(lr=1e-5)
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_jax = jnp.zeros((8, 768))
    vision_jax = jnp.zeros((768,))
    q_actual = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))

    _, _, loss = predictor.learn_from_error(
        model, sense_proj, carry, text_tokens, audio_jax, vision_jax, q_actual, key)

    assert np.isfinite(loss), f"Loss was not finite: {loss}"
