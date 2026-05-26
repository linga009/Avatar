"""Tests for Critical Order-Parameter Cognition engine."""
import math
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics

_CFG = Halo3Config()


def test_cop_init():
    cop = CriticalDynamics(_CFG)
    assert cop._tick == 0
    assert cop._chi_max == 1.0


def test_cop_observe_returns_dict():
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    result = cop.observe(
        r_mean=0.5, r_a=0.7, r_c=0.3,
        fe_delta=-0.01, K=0.3, theta=theta,
    )
    assert "chi" in result
    assert "tau" in result
    assert "unity" in result
    assert "gap" in result
    assert "K_new" in result
    assert "f_dot" in result
    assert "T_body" in result


def test_cop_chi_in_range():
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for i in range(25):
        r = 0.5 + 0.1 * math.sin(i * 0.3)
        result = cop.observe(r_mean=r, r_a=0.7, r_c=0.3,
                             fe_delta=-0.01, K=0.3, theta=theta)
    assert 0.0 <= result["chi"] <= 1.0


def test_cop_tau_in_range():
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for i in range(25):
        result = cop.observe(r_mean=0.5, r_a=0.7, r_c=0.3,
                             fe_delta=-0.01, K=0.3, theta=theta)
    assert 0.0 <= result["tau"] <= 1.0


def test_cop_soc_controller_undercoupled():
    """When r < 0.5 (undercoupled), SOC should increase K."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for i in range(10):
        r = 0.3 + 0.05 * math.sin(i)  # varies around 0.3, always < 0.5
        cop.observe(r_mean=r, r_a=0.4, r_c=0.2,
                    fe_delta=-0.01, K=0.3, theta=theta)
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert result["K_new"] > 0.3


def test_cop_soc_controller_overcoupled():
    """When r > 0.5 (overcoupled), SOC should decrease K."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for i in range(10):
        r = 0.7 + 0.05 * math.sin(i)  # varies around 0.7, always > 0.5
        cop.observe(r_mean=r, r_a=0.8, r_c=0.6,
                    fe_delta=-0.01, K=0.3, theta=theta)
    result = cop.observe(r_mean=0.7, r_a=0.8, r_c=0.6,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert result["K_new"] < 0.3


def test_cop_soc_controller_clamped():
    """K should never go below K_min or above K_max."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for _ in range(10):
        cop.observe(r_mean=0.5, r_a=0.7, r_c=0.3,
                    fe_delta=-0.01, K=0.01, theta=theta)
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.01, K=0.01, theta=theta)
    assert result["K_new"] >= _CFG.cop_K_min


def test_cop_warmup_disables_soc():
    """During warmup (first cop_warmup ticks), K should not change."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert result["K_new"] == 0.3


def test_cop_unity_in_range():
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for _ in range(10):
        result = cop.observe(r_mean=0.5, r_a=0.7, r_c=0.3,
                             fe_delta=-0.01, K=0.3, theta=theta)
    assert 0.0 <= result["unity"] <= 1.0 + 1e-5
    assert 0.0 <= result["gap"] <= 1.0 + 1e-5


def test_cop_t_body():
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    result = cop.observe(r_mean=0.5, r_a=0.8, r_c=0.2,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert abs(result["T_body"] - 0.6) < 0.01
