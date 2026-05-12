"""Tests for Halo3Config."""
import pytest
from halo3.config import Halo3Config

def test_default_instantiation():
    cfg = Halo3Config()
    assert cfg.d_model == 2048
    assert cfg.n_layers == 60
    assert cfg.d_state == 256
    assert cfg.vocab_size == 16000
    assert cfg.n_layers == 48
    assert cfg.n_clusters == 32
    assert cfg.mera_bond_dim == 64
    assert cfg.d_state == 128
    assert cfg.kuramoto_dt == 0.1

def test_frozen():
    cfg = Halo3Config()
    with pytest.raises(Exception):
        cfg.d_model = 512

def test_n_heads_d_head_consistency():
    cfg = Halo3Config()
    assert cfg.n_heads * cfg.d_head == cfg.d_model

def test_invalid_heads_raises():
    with pytest.raises(ValueError, match="n_heads"):
        Halo3Config(n_heads=3, d_head=128)

def test_small_config():
    cfg = Halo3Config(
        d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
        d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
        n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
        mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
        meta_n_hidden=4, meta_n_actions=2, meta_k=3,
        max_cache=8, island_size=4,
    )
    assert cfg.d_model == 64
