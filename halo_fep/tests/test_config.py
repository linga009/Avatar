import pytest
from halo_fep.config import HaloFEPConfig

def test_config_instantiates():
    cfg = HaloFEPConfig()
    assert cfg.d_model == 256
    assert cfg.n_agents == 256
    assert cfg.n_tokens == 2

def test_config_frozen():
    cfg = HaloFEPConfig()
    with pytest.raises(AttributeError):
        cfg.d_model = 512

def test_config_n_agents_divisible_by_coarse_k():
    cfg = HaloFEPConfig()
    assert cfg.n_agents % cfg.coarse_k == 0

def test_config_invalid_n_agents_raises():
    with pytest.raises(ValueError):
        HaloFEPConfig(n_agents=7, coarse_k=4)

def test_config_invalid_n_tokens_raises():
    with pytest.raises(ValueError):
        HaloFEPConfig(n_tokens=0)
