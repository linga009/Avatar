"""Integration test — full tick pipeline with spectral senses."""
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
        fno_vision_modes=4, codebook_size=8, codebook_dim=16,
        n_audio_tokens=2, n_vision_tokens=2,
    )


def test_full_tick_with_senses():
    """Simulate one full tick: perceive -> inject senses -> physics -> learn."""
    from halo3.model import Halo3Model, halo3_step
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule
    from halo3.senses.sensory_stats import SensoryStatistics

    cfg = _small_cfg()
    key = jax.random.PRNGKey(42)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)
    sensory_stats = SensoryStatistics(
        audio_tokens=cfg.n_audio_tokens,
        vision_tokens=cfg.n_vision_tokens,
        codebook_size=cfg.codebook_size)
    predictor = PredictiveProcessor(lr=1e-5)

    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_raw = jax.random.normal(key, (32000,))
    vision_raw = jax.random.normal(key, (224, 224, 3))

    # Inject senses
    tokens, sense_info = sense_module.process_and_inject(text_tokens, audio_raw, vision_raw)
    assert tokens.shape == (cfg.n_tokens, cfg.d_model)

    # Update sensory stats
    sensory_stats.update(sense_info["audio_indices"], sense_info["vision_indices"])
    pfc_line = sensory_stats.format_for_pfc()
    assert "audio" in pfc_line

    # Physics step
    key, sk = jax.random.split(key)
    carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)

    # Learn
    key, lk = jax.random.split(key)
    model, sense_module, loss, learn_info = predictor.learn_from_error(
        model, sense_module, carry, text_tokens, audio_raw, vision_raw, q_data, lk)

    assert np.isfinite(loss)
    assert "audio_indices" in learn_info
    assert sense_module.has_decoders  # still in critical period


def test_zero_input_tick():
    """Simulate tick with no capture agent running (all zeros)."""
    from halo3.model import Halo3Model, halo3_step
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule

    cfg = _small_cfg()
    key = jax.random.PRNGKey(0)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)
    predictor = PredictiveProcessor(lr=1e-5)

    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_raw = jnp.zeros((32000,))
    vision_raw = jnp.zeros((224, 224, 3))

    tokens, info = sense_module.process_and_inject(text_tokens, audio_raw, vision_raw)
    assert tokens.shape == (cfg.n_tokens, cfg.d_model)
    assert bool(jnp.all(jnp.isfinite(tokens)))

    key, sk = jax.random.split(key)
    carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)
    assert bool(jnp.all(jnp.isfinite(q_final)))
