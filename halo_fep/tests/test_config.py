from halo_fep.config import HaloFEPConfig

def test_config_instantiates():
    cfg = HaloFEPConfig()
    assert cfg.d_model == 256
    assert cfg.n_agents == 256
    assert cfg.n_tokens == 2

def test_config_frozen():
    import pytest
    cfg = HaloFEPConfig()
    with pytest.raises(Exception):
        cfg.d_model = 512

def test_config_n_agents_divisible_by_coarse_k():
    cfg = HaloFEPConfig()
    assert cfg.n_agents % cfg.coarse_k == 0
