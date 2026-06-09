"""Tests for SOC controller ablation."""
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics
import numpy as np


def _make_theta():
    import jax.numpy as jnp
    return jnp.zeros((128, 64))


def test_soc_disabled_k_unchanged():
    """With SOC disabled, K values should not change after observe()."""
    cfg = Halo3Config(disable_soc_controller=True)
    cop = CriticalDynamics(cfg)
    # Run past warmup
    for i in range(10):
        cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    result = cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                         K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    assert abs(result["K_aa"] - 0.5) < 1e-6, f"K_aa changed: {result['K_aa']}"
    assert abs(result["K_cc"] - 0.5) < 1e-6, f"K_cc changed: {result['K_cc']}"
    assert abs(result["K_cross"] - 0.25) < 1e-6, f"K_cross changed: {result['K_cross']}"


def test_soc_enabled_k_changes():
    """With SOC enabled (default), K should change after warmup."""
    cfg = Halo3Config(disable_soc_controller=False)
    cop = CriticalDynamics(cfg)
    for i in range(10):
        cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    result = cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                         K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    # r=0.3 < 0.5, so controller should increase K
    assert result["K_aa"] > 0.5 or result["K_cc"] > 0.5, "K should increase when r < 0.5"


def test_soc_disabled_avalanches_still_detected():
    """Avalanche detection runs even with SOC disabled."""
    cfg = Halo3Config(disable_soc_controller=True)
    cop = CriticalDynamics(cfg)
    # Drive r above then below threshold to create avalanche
    for i in range(20):
        cop.observe(r_mean=0.6, r_a=0.3, r_c=0.3, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    # Drop r — should start avalanche
    for i in range(5):
        cop.observe(r_mean=0.2, r_a=0.1, r_c=0.1, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    # Rise back — should end avalanche
    for i in range(5):
        result = cop.observe(r_mean=0.6, r_a=0.3, r_c=0.3, fe_delta=0.0,
                             K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    assert result["avalanche"]["n_total"] >= 1, "Avalanche should be detected even with SOC off"
