# halo_fep/tests/test_config.py
"""Tests for HaloFEPConfig — verifies all __post_init__ validation rules.

Covers
------
- Valid default config initialises without error.
- n_heads * d_head != d_model raises ValueError (new validation).
- flow_steps > n_layers raises ValueError (new validation).
- n_agents not divisible by coarse_k raises ValueError.
- n_tokens < 1 raises ValueError.
- wake_threshold <= 0 raises ValueError.
- tick_interval <= 0 raises ValueError.
- n_hidden < 1, n_obs < 1, n_actions < 1, tau < 1 raise ValueError.
- ewc_lambda < 0, per_alpha > 1, per_beta > 1 raise ValueError.
- n_tokens=32 is the correct default (not 2).
"""
import pytest
from halo_fep.config import HaloFEPConfig


# -------------------------------------------------------------------------
# Default config must be valid
# -------------------------------------------------------------------------

def test_default_config_valid():
    """HaloFEPConfig() with no arguments must succeed."""
    cfg = HaloFEPConfig()
    assert cfg is not None


def test_default_n_tokens_is_32():
    """n_tokens must default to 32 (production default; old default was 2)."""
    cfg = HaloFEPConfig()
    assert cfg.n_tokens == 32, (
        f"Expected n_tokens=32 (production default), got {cfg.n_tokens}. "
        "The old benchmark-only default of 2 has been retired."
    )


# -------------------------------------------------------------------------
# Attention head dimension validation (new)
# -------------------------------------------------------------------------

def test_n_heads_d_head_consistency():
    """n_heads * d_head == d_model must hold."""
    # Valid: 4 * 64 = 256
    cfg = HaloFEPConfig(n_heads=4, d_head=64, d_model=256)
    assert cfg is not None


def test_n_heads_d_head_mismatch_raises():
    """n_heads * d_head != d_model must raise ValueError."""
    with pytest.raises(ValueError, match="n_heads.*d_head.*d_model"):
        HaloFEPConfig(n_heads=5, d_head=64, d_model=256)  # 5*64=320 != 256


# -------------------------------------------------------------------------
# Flow steps validation (new)
# -------------------------------------------------------------------------

def test_flow_steps_leq_n_layers():
    """flow_steps == n_layers should be valid."""
    HaloFEPConfig(flow_steps=8, n_layers=8)


def test_flow_steps_exceeds_n_layers_raises():
    """flow_steps > n_layers must raise ValueError."""
    with pytest.raises(ValueError, match="flow_steps.*n_layers"):
        HaloFEPConfig(flow_steps=9, n_layers=8)


# -------------------------------------------------------------------------
# Swarm validation
# -------------------------------------------------------------------------

def test_n_agents_not_divisible_by_coarse_k_raises():
    """n_agents must be divisible by coarse_k."""
    with pytest.raises(ValueError, match="n_agents.*coarse_k"):
        HaloFEPConfig(n_agents=100, coarse_k=16)


# -------------------------------------------------------------------------
# Token / heartbeat validation
# -------------------------------------------------------------------------

def test_n_tokens_zero_raises():
    with pytest.raises(ValueError, match="n_tokens"):
        HaloFEPConfig(n_tokens=0)


def test_wake_threshold_zero_raises():
    with pytest.raises(ValueError, match="wake_threshold"):
        HaloFEPConfig(wake_threshold=0.0)


def test_wake_threshold_negative_raises():
    with pytest.raises(ValueError, match="wake_threshold"):
        HaloFEPConfig(wake_threshold=-1.0)


def test_tick_interval_zero_raises():
    with pytest.raises(ValueError, match="tick_interval"):
        HaloFEPConfig(tick_interval=0)


# -------------------------------------------------------------------------
# FEP dimension validation
# -------------------------------------------------------------------------

def test_n_hidden_zero_raises():
    with pytest.raises(ValueError, match="n_hidden"):
        HaloFEPConfig(n_hidden=0)


def test_n_obs_zero_raises():
    with pytest.raises(ValueError, match="n_obs"):
        HaloFEPConfig(n_obs=0)


def test_n_actions_zero_raises():
    with pytest.raises(ValueError, match="n_actions"):
        HaloFEPConfig(n_actions=0)


def test_tau_zero_raises():
    with pytest.raises(ValueError, match="tau"):
        HaloFEPConfig(tau=0)


# -------------------------------------------------------------------------
# Learning / regularisation bounds
# -------------------------------------------------------------------------

def test_ewc_lambda_negative_raises():
    with pytest.raises(ValueError, match="ewc_lambda"):
        HaloFEPConfig(ewc_lambda=-0.1)


def test_per_alpha_above_one_raises():
    with pytest.raises(ValueError, match="per_alpha"):
        HaloFEPConfig(per_alpha=1.1)


def test_per_beta_above_one_raises():
    with pytest.raises(ValueError, match="per_beta"):
        HaloFEPConfig(per_beta=1.1)


def test_per_alpha_zero_valid():
    """per_alpha=0 (uniform sampling) must be valid."""
    HaloFEPConfig(per_alpha=0.0)


def test_per_beta_zero_valid():
    """per_beta=0 (no IS correction) must be valid."""
    HaloFEPConfig(per_beta=0.0)
