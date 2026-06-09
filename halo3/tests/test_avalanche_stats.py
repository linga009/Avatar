"""Tests for rigorous avalanche statistics."""
import numpy as np
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics


def _make_power_law_samples(n, tau, x_min=0.01, seed=42):
    """Generate synthetic power-law distributed samples."""
    rng = np.random.RandomState(seed)
    u = rng.uniform(0, 1, n)
    return x_min * u ** (-1.0 / (tau - 1.0))


def _make_exponential_samples(n, rate=1.0, seed=42):
    """Generate synthetic exponential samples."""
    rng = np.random.RandomState(seed)
    return rng.exponential(1.0 / rate, n)


def test_rigorous_returns_none_below_50():
    """Returns None when n < 50."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_sizes = [0.1] * 30
    cop._avalanche_durations = [2] * 30
    result = cop.avalanche_stats_rigorous
    assert result is None


def test_ks_power_law_accepted():
    """Synthetic power-law data should have p > 0.05."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    sizes = _make_power_law_samples(100, tau=1.5).tolist()
    durations = [max(1, int(s * 10)) for s in sizes]
    cop._avalanche_sizes = sizes
    cop._avalanche_durations = durations
    result = cop.avalanche_stats_rigorous
    assert result is not None
    assert result["ks_p_size"] > 0.05, f"Power-law data should not be rejected, p={result['ks_p_size']}"


def test_ks_exponential_rejected():
    """Synthetic exponential data should have p < 0.2."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    sizes = _make_exponential_samples(100).tolist()
    durations = [max(1, int(s * 5)) for s in sizes]
    cop._avalanche_sizes = sizes
    cop._avalanche_durations = durations
    result = cop.avalanche_stats_rigorous
    assert result is not None
    assert result["ks_p_size"] < 0.2, f"Exponential data should be rejected or marginal, p={result['ks_p_size']}"


def test_bootstrap_ci_contains_true():
    """Bootstrap 95% CI should contain the true tau for synthetic data."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    true_tau = 1.5
    sizes = _make_power_law_samples(200, tau=true_tau, seed=123).tolist()
    durations = [max(1, int(s * 10)) for s in sizes]
    cop._avalanche_sizes = sizes
    cop._avalanche_durations = durations
    result = cop.avalanche_stats_rigorous
    assert result is not None
    ci_lo, ci_hi = result["tau_ci"]
    assert ci_lo <= true_tau <= ci_hi, f"True tau {true_tau} not in CI [{ci_lo}, {ci_hi}]"


def test_scaling_relation():
    """gamma = (tau-1)/(alpha-1) should be computed."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_sizes = _make_power_law_samples(100, tau=1.5).tolist()
    cop._avalanche_durations = [max(1, int(s * 10)) for s in cop._avalanche_sizes]
    result = cop.avalanche_stats_rigorous
    assert result is not None
    assert "gamma" in result
    assert isinstance(result["gamma"], float)
