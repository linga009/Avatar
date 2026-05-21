"""Test SensoryStatistics — codebook activation tracking for PFC."""
import jax.numpy as jnp
import numpy as np
import json
import pytest


def test_update_and_flux():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=8, vision_tokens=4, codebook_size=32)
    stats.update(jnp.array([0, 1, 2, 3, 4, 5, 6, 7]),
                 jnp.array([0, 1, 2, 3]))
    assert stats.audio_flux == 0
    stats.update(jnp.array([0, 1, 2, 3, 4, 5, 6, 8]),
                 jnp.array([0, 1, 2, 4]))
    assert stats.audio_flux == 1
    assert stats.vision_flux == 1


def test_stability_tracking():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=8, vision_tokens=4, codebook_size=32)
    codes_a = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    codes_v = jnp.array([0, 1, 2, 3])
    for _ in range(5):
        stats.update(codes_a, codes_v)
    assert stats.audio_stability >= 4
    assert stats.vision_stability >= 4


def test_novelty_computation():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=2, vision_tokens=2, codebook_size=32)
    for _ in range(20):
        stats.update(jnp.array([0, 1]), jnp.array([0, 1]))
    low_novelty = stats.audio_novelty
    stats.update(jnp.array([30, 31]), jnp.array([30, 31]))
    high_novelty = stats.audio_novelty
    assert high_novelty > low_novelty


def test_cross_modal_binding():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=2, vision_tokens=2, codebook_size=32)
    for _ in range(20):
        stats.update(jnp.array([0, 1]), jnp.array([2, 3]))
    familiar = stats.cross_modal_binding
    stats.update(jnp.array([10, 11]), jnp.array([20, 21]))
    novel = stats.cross_modal_binding
    assert familiar > novel


def test_format_for_pfc():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=8, vision_tokens=4, codebook_size=32)
    stats.update(jnp.array([0, 1, 2, 3, 4, 5, 6, 7]),
                 jnp.array([0, 1, 2, 3]))
    line = stats.format_for_pfc()
    assert "audio" in line
    assert "vision" in line
    assert "binding" in line


def test_save_load_roundtrip(tmp_path):
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=8, vision_tokens=4, codebook_size=32)
    stats.update(jnp.array([0, 1, 2, 3, 4, 5, 6, 7]),
                 jnp.array([0, 1, 2, 3]))
    path = str(tmp_path / "sensory_stats.json")
    stats.save(path)
    stats2 = SensoryStatistics(audio_tokens=8, vision_tokens=4, codebook_size=32)
    stats2.load(path)
    assert stats2.audio_stability == stats.audio_stability
