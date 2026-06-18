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
        fe_delta=-0.01, K_aa=0.3, K_cc=0.3, K_cross=0.15, theta=theta,
    )
    assert "chi" in result
    assert "tau" in result
    assert "unity" in result
    assert "gap" in result
    assert "K_new" in result
    assert "K_aa" in result
    assert "K_cc" in result
    assert "K_cross" in result
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
    """When r < 0.5 (undercoupled), SOC should increase K_aa and K_cc on average."""
    import random as _rng
    _rng.seed(42)  # deterministic noise for test
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    # Start within block-specific bounds: analytical [0.02, 0.40], creative [0.20, 2.00]
    K_aa, K_cc, K_cross = 0.05, 0.80, 0.15
    for i in range(30):  # enough ticks for proportional signal to dominate noise
        r = 0.3 + 0.05 * math.sin(i)  # varies around 0.3, always < 0.5
        result = cop.observe(r_mean=r, r_a=0.4, r_c=0.2,
                    fe_delta=-0.01, K_aa=K_aa, K_cc=K_cc, K_cross=K_cross,
                    theta=theta)
        K_aa, K_cc, K_cross = result["K_aa"], result["K_cc"], result["K_cross"]
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.01, K_aa=K_aa, K_cc=K_cc, K_cross=K_cross,
                         theta=theta)
    assert result["K_aa"] > 0.05  # analytical undercoupled, should increase
    assert result["K_cc"] > 0.80  # creative very undercoupled, should increase more


def test_cop_soc_controller_overcoupled():
    """When r > 0.5 (overcoupled), SOC should decrease K_aa and K_cc on average."""
    import random as _rng
    _rng.seed(43)  # deterministic noise for test
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    # Start within block-specific bounds: analytical [0.02, 0.40], creative [0.20, 2.00]
    K_aa, K_cc, K_cross = 0.30, 1.80, 0.5
    for i in range(30):  # enough ticks for proportional signal to dominate noise
        r = 0.7 + 0.05 * math.sin(i)  # varies around 0.7, always > 0.5
        result = cop.observe(r_mean=r, r_a=0.8, r_c=0.6,
                    fe_delta=-0.01, K_aa=K_aa, K_cc=K_cc, K_cross=K_cross,
                    theta=theta)
        K_aa, K_cc, K_cross = result["K_aa"], result["K_cc"], result["K_cross"]
    result = cop.observe(r_mean=0.7, r_a=0.8, r_c=0.6,
                         fe_delta=-0.01, K_aa=K_aa, K_cc=K_cc, K_cross=K_cross,
                         theta=theta)
    assert result["K_aa"] < 0.30  # analytical overcoupled, should decrease
    assert result["K_cc"] < 1.80  # creative overcoupled, should decrease


def test_cop_soc_controller_no_floor_lock():
    """K_aa must NOT stay collapsed during sustained overcoupling.

    Regression test for the June 14-17 2026 incident where K_aa was stuck
    at 0.02-0.09 for 3 days while K_cc maxed at 1.8-2.0, causing 100% DARK.
    The floor-lock recovery should keep K_aa oscillating in a healthy range.
    Brief transient dips are OK; sustained collapse is not.
    """
    import random as _rng
    _rng.seed(99)
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    K_aa, K_cc, K_cross = 0.20, 1.00, 0.5
    K_aa_history = []
    for i in range(200):  # simulate ~200 ticks of persistent r_a > 0.5
        r_a = 0.7 + 0.05 * math.sin(i * 0.3)  # always > 0.5
        r_c = 0.3 + 0.05 * math.sin(i * 0.2)  # always < 0.5
        r = 0.5 * (r_a + r_c)
        result = cop.observe(r_mean=r, r_a=r_a, r_c=r_c,
                             fe_delta=-0.01, K_aa=K_aa, K_cc=K_cc, K_cross=K_cross,
                             theta=theta)
        K_aa, K_cc, K_cross = result["K_aa"], result["K_cc"], result["K_cross"]
        K_aa_history.append(K_aa)

    K_aa_range = _CFG.cop_K_max_aa - _CFG.cop_K_min_aa
    # Average K_aa over last 100 ticks should be well above floor
    avg_last_100 = sum(K_aa_history[-100:]) / 100
    assert avg_last_100 > _CFG.cop_K_min_aa + 0.15 * K_aa_range, \
        f"K_aa sustained collapse: avg last 100 = {avg_last_100:.4f}"
    # K_aa should not spend >20% of time near the floor (bottom 10% of range)
    floor_threshold = _CFG.cop_K_min_aa + 0.10 * K_aa_range
    floor_ticks = sum(1 for k in K_aa_history if k < floor_threshold)
    assert floor_ticks < 40, \
        f"K_aa near floor {floor_ticks}/200 ticks — recovery too weak"
    # Final K_aa should have recovered, not be stuck
    assert K_aa > _CFG.cop_K_min_aa + 0.10 * K_aa_range, \
        f"K_aa ended at {K_aa:.4f} — still collapsed"


def test_cop_soc_controller_clamped():
    """K should respect block-specific bounds."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    for _ in range(10):
        cop.observe(r_mean=0.5, r_a=0.7, r_c=0.3,
                    fe_delta=-0.01, K_aa=0.01, K_cc=0.01, K_cross=0.01,
                    theta=theta)
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.01, K_aa=0.01, K_cc=0.01, K_cross=0.01,
                         theta=theta)
    # Block-specific bounds: analytical [0.02, 0.40], creative [0.20, 2.00]
    assert result["K_aa"] >= _CFG.cop_K_min_aa
    assert result["K_aa"] <= _CFG.cop_K_max_aa
    assert result["K_cc"] >= _CFG.cop_K_min_cc
    assert result["K_cc"] <= _CFG.cop_K_max_cc
    assert result["K_cross"] >= _CFG.cop_K_min


def test_cop_warmup_disables_soc():
    """During warmup (first cop_warmup ticks), K values should not change."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    result = cop.observe(r_mean=0.3, r_a=0.4, r_c=0.2,
                         fe_delta=-0.01, K_aa=0.10, K_cc=1.0, K_cross=0.15,
                         theta=theta)
    assert result["K_aa"] == 0.10
    assert result["K_cc"] == 1.0
    assert result["K_cross"] == 0.15


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


def test_cop_chi_driven_system_lower():
    """Driven system (obs_norm correlated with r) should have chi <= free chi + 0.05."""
    cop_driven = CriticalDynamics(_CFG)
    cop_free = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))

    for i in range(30):
        r = 0.5 + 0.2 * math.sin(i * 0.3)
        obs_driven = 0.5 + 0.2 * math.sin(i * 0.3)  # in sync with r
        obs_free = 0.0  # constant

        res_driven = cop_driven.observe(
            r_mean=r, r_a=0.5, r_c=0.5, fe_delta=-0.01,
            K_aa=0.3, K_cc=0.3, K_cross=0.15,
            theta=theta, obs_norm=obs_driven,
        )
        res_free = cop_free.observe(
            r_mean=r, r_a=0.5, r_c=0.5, fe_delta=-0.01,
            K_aa=0.3, K_cc=0.3, K_cross=0.15,
            theta=theta, obs_norm=obs_free,
        )

    assert res_driven["chi"] <= res_free["chi"] + 0.05


def test_cop_harada_sasa_still_bounded():
    """chi must stay in [0.0, 1.0] even with correlated r and obs_norm."""
    cop = CriticalDynamics(_CFG)
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))

    for i in range(50):
        r = 0.5 + 0.3 * math.sin(i * 0.2)
        obs = 0.5 + 0.3 * math.cos(i * 0.2)  # correlated but phase-shifted
        cop.observe(
            r_mean=r, r_a=0.5, r_c=0.5, fe_delta=-0.01,
            K_aa=0.3, K_cc=0.3, K_cross=0.15,
            theta=theta, obs_norm=obs,
        )

    result = cop.observe(
        r_mean=0.5, r_a=0.5, r_c=0.5, fe_delta=-0.01,
        K_aa=0.3, K_cc=0.3, K_cross=0.15,
        theta=theta, obs_norm=0.3,
    )
    assert 0.0 <= result["chi"] <= 1.0


def test_cop_f_thermo_in_output():
    """F_thermo should appear in output dict as a float when H_mean is provided."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(42),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for i in range(10):
        result = cop.observe(
            r_mean=0.5 + 0.05 * math.sin(i),
            r_a=0.6, r_c=0.4,
            fe_delta=-0.01,
            K_aa=0.3, K_cc=0.3, K_cross=0.15,
            theta=theta,
            H_mean=1.5,
        )
    assert "F_thermo" in result
    assert isinstance(result["F_thermo"], float)


def test_cop_f_thermo_none_without_H():
    """F_thermo should be None when H_mean is not provided."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(7),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    for i in range(10):
        result = cop.observe(
            r_mean=0.5, r_a=0.6, r_c=0.4,
            fe_delta=-0.01,
            K_aa=0.3, K_cc=0.3, K_cross=0.15,
            theta=theta,
        )
    assert result["F_thermo"] is None


# --- Avalanche detection tests ---

def test_cop_avalanche_in_output():
    """observe() should return an avalanche dict."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(0),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    result = cop.observe(
        r_mean=0.5, r_a=0.6, r_c=0.4,
        fe_delta=-0.01, theta=theta,
    )
    assert "avalanche" in result
    assert "n_total" in result["avalanche"]
    assert "in_avalanche" in result["avalanche"]


def test_cop_avalanche_detects_dip():
    """A sequence with r dipping below threshold should produce an avalanche."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(1),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi

    # Feed stable r=0.5 for 20 ticks to establish threshold
    for _ in range(20):
        cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                    fe_delta=0.0, theta=theta)

    # Dip below threshold for 5 ticks
    for _ in range(5):
        cop.observe(r_mean=0.3, r_a=0.3, r_c=0.3,
                    fe_delta=0.0, theta=theta)

    # Return above threshold — avalanche should end
    result = cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                         fe_delta=0.0, theta=theta)

    assert result["avalanche"]["n_total"] >= 1
    assert result["avalanche"]["just_ended"] is True


def test_cop_avalanche_no_false_positives():
    """Constant r should produce zero avalanches."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(2),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi

    for _ in range(50):
        cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                    fe_delta=0.0, theta=theta)

    assert cop.avalanche_stats["n"] == 0


def test_cop_avalanche_stats_exponents():
    """With enough avalanches, stats should return exponent estimates."""
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(3),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi

    # Generate 30 avalanches: high-low-high cycles
    for cycle in range(30):
        # Above threshold
        for _ in range(5):
            cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                        fe_delta=0.0, theta=theta)
        # Below threshold (varying depth for diverse sizes)
        dip = 0.3 - 0.01 * (cycle % 10)
        for _ in range(2 + cycle % 4):
            cop.observe(r_mean=dip, r_a=dip, r_c=dip,
                        fe_delta=0.0, theta=theta)

    # End last avalanche
    cop.observe(r_mean=0.5, r_a=0.5, r_c=0.5,
                fe_delta=0.0, theta=theta)

    stats = cop.avalanche_stats
    assert stats["n"] >= 20
    assert stats["tau_est"] is not None
    assert stats["tau_est"] > 1.0  # power-law exponent > 1
    assert stats["alpha_est"] is not None
    assert stats["mean_size"] > 0
    assert stats["mean_duration"] > 0


# --- Post-dream reset tests ---

def test_cop_post_dream_chi_nonzero():
    """post_dream_reset must NOT crash chi to zero.

    Regression test for June 18 2026: identical pre-fill values gave
    Var(r)=0, so chi_raw=N*0=0. Chi stayed at ~0 for 40+ ticks, killing
    the SOC controller and preventing consciousness ignition.
    """
    cop = CriticalDynamics(_CFG)
    theta = jax.random.uniform(jax.random.PRNGKey(77),
                               (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    # Build up some history first
    for i in range(20):
        r = 0.5 + 0.1 * math.sin(i * 0.3)
        cop.observe(r_mean=r, r_a=0.6, r_c=0.4,
                    fe_delta=-0.01, K_aa=0.2, K_cc=1.0, K_cross=0.5,
                    theta=theta)
    # Dream reset
    cop.post_dream_reset(terminal_r=0.55)
    # First post-dream tick should have nonzero chi_raw
    result = cop.observe(r_mean=0.55, r_a=0.6, r_c=0.5,
                         fe_delta=-0.01, K_aa=0.2, K_cc=1.0, K_cross=0.5,
                         theta=theta)
    assert result["chi_raw"] > 0.1, \
        f"chi_raw={result['chi_raw']:.4f} — post-dream pre-fill has no variance"
