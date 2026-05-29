from experiments.configs import get_experiment_config, EXPERIMENTS


def test_all_experiments_defined():
    assert len(EXPERIMENTS) == 6
    assert "full_avatar" in EXPERIMENTS
    assert "no_cop" in EXPERIMENTS
    assert "no_senses" in EXPERIMENTS
    assert "no_dreams" in EXPERIMENTS
    assert "no_bohmian_q" in EXPERIMENTS
    assert "transformer_baseline" in EXPERIMENTS


def test_full_avatar_is_default():
    cfg = get_experiment_config("full_avatar")
    assert cfg.disable_cop is False
    assert cfg.disable_senses is False
    assert cfg.disable_dreams is False
    assert cfg.disable_quantum_potential is False
    assert cfg.is_transformer_baseline is False


def test_no_cop_config():
    cfg = get_experiment_config("no_cop")
    assert cfg.disable_cop is True
    assert cfg.disable_senses is False


def test_no_dreams_config():
    cfg = get_experiment_config("no_dreams")
    assert cfg.disable_dreams is True
