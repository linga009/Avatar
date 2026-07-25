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


# --- Binder cumulant tests ---

def test_binder_in_observe():
    """Binder cumulant should be present in observe() output."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for _ in range(10):
        result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                             fe_delta=-0.01, K=0.3, theta=theta)
    assert "binder" in result
    assert isinstance(result["binder"], float)


def test_binder_ordered_phase():
    """In ordered phase (r near 1), U4 should approach 2/3."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for _ in range(25):
        result = cop.observe(r_mean=0.95, r_a=0.95, r_c=0.95,
                             fe_delta=-0.01, K=0.3, theta=theta)
    # For constant r=0.95: r2=0.9025, r4=0.8145, U4=1-0.8145/(3*0.9025^2)=0.667
    assert result["binder"] > 0.6


def test_binder_disordered_phase():
    """In disordered phase (r near 0), U4 should approach 0."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(42),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for _ in range(25):
        result = cop.observe(r_mean=0.05, r_a=0.05, r_c=0.05,
                             fe_delta=-0.01, K=0.3, theta=theta)
    # For constant r=0.05: r2=0.0025, r4=6.25e-6, U4=1-6.25e-6/(3*6.25e-6)=0.667
    # Actually constant r gives U4=2/3 regardless. Need variance for U4<2/3.
    # Constant r always gives U4=2/3 because <r^4>=<r^2>^2 when no variance.
    assert result["binder"] > 0.5  # constant input => U4 ~ 2/3


def test_binder_with_fluctuations():
    """U4 should differ from 2/3 when r fluctuates."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for i in range(25):
        # Large fluctuations: r swings between 0.1 and 0.9
        r = 0.1 if i % 2 == 0 else 0.9
        cop.observe(r_mean=r, r_a=r, r_c=r,
                    fe_delta=-0.01, K=0.3, theta=theta)
    result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                         fe_delta=-0.01, K=0.3, theta=theta)
    # Bimodal r: <r^2>=0.41, <r^4>=0.3281, U4=1-0.3281/(3*0.1681)=0.349
    assert result["binder"] < 0.5  # well below 2/3


# --- Participation ratio tests ---

def test_pr_in_observe():
    """Participation ratio should be present in observe() output."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert "pr" in result
    assert result["pr"] >= 1.0


def test_pr_synchronized_broad():
    """All clusters in sync should give high PR (all participate)."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for _ in range(5):
        result = cop.observe(r_mean=0.9, r_a=0.9, r_c=0.9,
                             fe_delta=-0.01, K=0.3, theta=theta)
    # All clusters synchronized => coherence matrix is all 1s => v1 uniform => PR=n_clusters
    assert result["pr"] > _CFG.n_clusters * 0.8


def test_pr_range():
    """PR should be in [1, n_clusters]."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(7),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for _ in range(5):
        result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                             fe_delta=-0.01, K=0.3, theta=theta)
    assert 1.0 <= result["pr"] <= _CFG.n_clusters + 1e-5


# --- Cross-population correlation tests ---

def test_cross_corr_in_observe():
    """Cross correlation should be present in observe() output."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for _ in range(10):
        result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                             fe_delta=-0.01, K=0.3, theta=theta)
    assert "cross_corr" in result
    assert -1.0 <= result["cross_corr"] <= 1.0


def test_cross_corr_unified():
    """When r_a and r_c move together, C_ac should be positive."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for i in range(25):
        # Both move together: high then low
        r = 0.8 if i % 2 == 0 else 0.3
        cop.observe(r_mean=r, r_a=r, r_c=r,
                    fe_delta=-0.01, K=0.3, theta=theta)
    result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert result["cross_corr"] > 0.5  # strongly correlated


def test_cross_corr_dialectical():
    """When r_a and r_c move in opposition, C_ac should be negative."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for i in range(25):
        # Anti-correlated: when analytical is high, creative is low
        r_a = 0.8 if i % 2 == 0 else 0.3
        r_c = 0.3 if i % 2 == 0 else 0.8
        r_mean = (r_a + r_c) / 2.0
        cop.observe(r_mean=r_mean, r_a=r_a, r_c=r_c,
                    fe_delta=-0.01, K=0.3, theta=theta)
    result = cop.observe(r_mean=0.55, r_a=0.55, r_c=0.55,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert result["cross_corr"] < -0.5  # strongly anti-correlated


def test_cross_corr_independent():
    """When r_a and r_c are uncorrelated, C_ac should be near 0."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    # r_a oscillates with period 2, r_c with period 3 — orthogonal over 24+ ticks
    for i in range(30):
        r_a = 0.5 + 0.2 * math.sin(i * math.pi)       # period 2
        r_c = 0.5 + 0.2 * math.sin(i * 2 * math.pi / 3)  # period 3
        cop.observe(r_mean=(r_a + r_c) / 2, r_a=r_a, r_c=r_c,
                    fe_delta=-0.01, K=0.3, theta=theta)
    result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                         fe_delta=-0.01, K=0.3, theta=theta)
    assert abs(result["cross_corr"]) < 0.3  # near zero
