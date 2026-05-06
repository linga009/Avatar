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

def test_new_training_fields_have_correct_defaults():
    cfg = HaloFEPConfig()
    assert cfg.ewc_lambda == 0.1
    assert cfg.per_alpha == 0.6
    assert cfg.per_beta == 0.4
    assert cfg.use_mesu is False
    assert cfg.mesu_eta == 0.01

def test_ewc_lambda_zero_disables_ewc():
    cfg = HaloFEPConfig(ewc_lambda=0.0)
    assert cfg.ewc_lambda == 0.0
