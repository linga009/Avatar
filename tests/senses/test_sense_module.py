"""Test SenseModule — full pipeline: FNO -> VQ-VAE -> injection."""
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


def test_sense_module_inject_shape():
    from halo3.senses.sense_module import SenseModule
    cfg = _small_cfg()
    key = jax.random.PRNGKey(0)
    sm = SenseModule(cfg, key=key)
    text_tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
    audio = jnp.zeros((32000,))
    vision = jnp.zeros((224, 224, 3))
    out, info = sm.process_and_inject(text_tokens, audio, vision)
    assert out.shape == (cfg.n_tokens, cfg.d_model)


def test_sense_module_returns_indices():
    from halo3.senses.sense_module import SenseModule
    cfg = _small_cfg()
    key = jax.random.PRNGKey(1)
    sm = SenseModule(cfg, key=key)
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio = jax.random.normal(key, (32000,))
    vision = jax.random.normal(key, (224, 224, 3))
    _, info = sm.process_and_inject(text_tokens, audio, vision)
    assert info["audio_indices"].shape == (cfg.n_audio_tokens,)
    assert info["vision_indices"].shape == (cfg.n_vision_tokens,)
    assert "commitment_loss" in info


def test_sense_module_zero_input_passthrough():
    from halo3.senses.sense_module import SenseModule
    cfg = _small_cfg()
    key = jax.random.PRNGKey(2)
    sm = SenseModule(cfg, key=key)
    text_tokens = jnp.ones((cfg.n_tokens, cfg.d_model))
    audio = jnp.zeros((32000,))
    vision = jnp.zeros((224, 224, 3))
    out, _ = sm.process_and_inject(text_tokens, audio, vision)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_sense_module_finite_output():
    from halo3.senses.sense_module import SenseModule
    cfg = _small_cfg()
    key = jax.random.PRNGKey(3)
    sm = SenseModule(cfg, key=key)
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio = jax.random.normal(key, (32000,))
    vision = jax.random.normal(key, (224, 224, 3))
    out, info = sm.process_and_inject(text_tokens, audio, vision)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert np.isfinite(float(info["commitment_loss"]))


def test_sense_module_has_decoder_initially():
    from halo3.senses.sense_module import SenseModule
    cfg = _small_cfg()
    key = jax.random.PRNGKey(4)
    sm = SenseModule(cfg, key=key)
    assert sm.has_decoders


def test_sense_module_reconstruction_loss():
    from halo3.senses.sense_module import SenseModule
    cfg = _small_cfg()
    key = jax.random.PRNGKey(5)
    sm = SenseModule(cfg, key=key)
    audio = jax.random.normal(key, (32000,))
    vision = jax.random.normal(key, (224, 224, 3))
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    _, info = sm.process_and_inject(text_tokens, audio, vision)
    recon_loss = sm.reconstruction_loss(audio, vision, info)
    assert np.isfinite(float(recon_loss))
    assert float(recon_loss) > 0.0


def test_sense_module_delete_decoders():
    from halo3.senses.sense_module import SenseModule, delete_decoders
    cfg = _small_cfg()
    key = jax.random.PRNGKey(6)
    sm = SenseModule(cfg, key=key)
    assert sm.has_decoders
    sm2 = delete_decoders(sm)
    assert not sm2.has_decoders
