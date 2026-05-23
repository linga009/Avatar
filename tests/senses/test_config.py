"""Test FNO/VQ-VAE config fields."""
from halo3.config import Halo3Config


def test_fno_config_defaults():
    cfg = Halo3Config()
    assert cfg.fno_hidden_dim == 64
    assert cfg.fno_n_layers == 4
    assert cfg.fno_audio_modes == 32
    assert cfg.fno_vision_modes == 16
    assert cfg.codebook_size_audio == 128
    assert cfg.codebook_size_vision == 64
    assert cfg.codebook_dim == 64
    assert cfg.codebook_ema_decay == 0.99
    assert cfg.commitment_beta == 0.25
    assert cfg.dead_code_threshold == 100
    assert cfg.n_audio_tokens == 16
    assert cfg.n_vision_tokens == 8
    assert cfg.critical_period_recon_weight == 0.5


def test_phase_b_config_defaults():
    cfg = Halo3Config()
    assert cfg.tts_mode == "espeak"
    assert cfg.tts_every_n == 3
    assert cfg.contrastive_tau == 0.07
    assert cfg.contrastive_weight == 0.3
    assert cfg.contrastive_maturation_threshold == 0.75
