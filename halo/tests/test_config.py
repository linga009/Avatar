# halo/tests/test_config.py
from halo.config import HaloConfig

def test_default_config():
    cfg = HaloConfig()
    assert cfg.d_model == 256
    assert cfg.d_boundary == 64
    assert cfg.d_head == 64
    assert cfg.n_heads == 4
    assert cfg.n_layers == 8
    assert cfg.delta_text == 1.0
    assert cfg.delta_image == 2.0
    assert cfg.max_cache == 128
    assert cfg.flow_steps == 4

def test_custom_config():
    cfg = HaloConfig(d_model=128, n_layers=4)
    assert cfg.d_model == 128
    assert cfg.n_layers == 4
    assert cfg.d_boundary == 64  # unchanged default
