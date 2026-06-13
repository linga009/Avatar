"""Smoke tests for experiment runner — import and metrics check."""
from experiments.configs import EXPERIMENTS, get_experiment_config, ExperimentConfig


def test_all_configs_loadable():
    """All 6 experiment configs load without error."""
    for name in EXPERIMENTS:
        cfg = get_experiment_config(name)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.name == name


def test_config_flags():
    """Key ablation flags are set correctly."""
    assert get_experiment_config("no_cop").disable_cop is True
    assert get_experiment_config("no_senses").disable_senses is True
    assert get_experiment_config("no_dreams").disable_dreams is True
    assert get_experiment_config("no_bohmian_q").disable_quantum_potential is True
    assert get_experiment_config("transformer_baseline").is_transformer_baseline is True
    assert get_experiment_config("full_avatar").disable_cop is False


def test_runner_importable():
    """experiment_runner module imports without error."""
    from experiments import experiment_runner
    assert hasattr(experiment_runner, "run_experiment")


import pytest


@pytest.mark.slow
def test_full_avatar_3_ticks():
    """Run full_avatar for 3 ticks — smoke test for the entire pipeline."""
    from experiments.configs import ExperimentConfig
    from experiments.experiment_runner import run_experiment

    exp = ExperimentConfig(
        name="smoke_test",
        n_ticks=3,
        description="3-tick smoke test",
    )
    csv_path = run_experiment(exp)
    assert csv_path.exists(), f"CSV not created at {csv_path}"

    # Check CSV has header + 3 data rows
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) >= 4, f"Expected 4+ lines (header + 3 ticks), got {len(lines)}"
