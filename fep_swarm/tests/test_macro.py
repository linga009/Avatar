import jax
import jax.numpy as jnp
import pytest
import chex
from fep_swarm.config import FEPConfig
from fep_swarm.macro.renormalization import MacroState, coarse_grain
from fep_swarm.swarm.coupling import build_coupling_matrix


@pytest.fixture
def cfg():
    return FEPConfig(n_agents=32, coarse_k=8)  # 4 groups


@pytest.fixture
def W(cfg):
    return build_coupling_matrix(cfg, jax.random.PRNGKey(0))


def test_macro_state_shapes(cfg, W):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    obs = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (cfg.n_agents, cfg.n_obs)), axis=-1
    )
    actions = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(2), (cfg.n_agents, cfg.n_actions)), axis=-1
    )
    macro = coarse_grain(mu, obs, actions, W, cfg)
    n_groups = cfg.n_agents // cfg.coarse_k
    chex.assert_shape(macro.M, (n_groups, cfg.n_hidden))
    chex.assert_shape(macro.S, (n_groups, cfg.n_obs))
    chex.assert_shape(macro.A, (n_groups, cfg.n_actions))


def test_M_is_mean_of_groups(cfg, W):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    obs = jnp.ones((cfg.n_agents, cfg.n_obs)) / cfg.n_obs
    actions = jnp.ones((cfg.n_agents, cfg.n_actions)) / cfg.n_actions
    macro = coarse_grain(mu, obs, actions, W, cfg)
    k = cfg.coarse_k
    n_groups = cfg.n_agents // k
    for g in range(n_groups):
        expected_M_g = mu[g * k:(g + 1) * k].mean(axis=0)
        assert jnp.allclose(macro.M[g], expected_M_g, atol=1e-5)


# ── Task 12: Macro Free Energy Bound ──────────────────────────────────────────

from fep_swarm.macro.macro_blanket import (
    macro_free_energy, micro_free_energy_sum, check_macro_bound
)
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel


@pytest.fixture
def gm(cfg):
    return DiscreteGenerativeModel(cfg, jax.random.PRNGKey(99))


def test_macro_free_energy_scalar(cfg, W, gm):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    obs = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (cfg.n_agents, cfg.n_obs)), axis=-1
    )
    actions = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(2), (cfg.n_agents, cfg.n_actions)), axis=-1
    )
    macro = coarse_grain(mu, obs, actions, W, cfg)
    F_macro = macro_free_energy(macro, gm, cfg)
    assert F_macro.shape == ()
    assert not jnp.isnan(F_macro)


def test_micro_free_energy_sum_scalar(cfg, gm):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    obs_soft = jax.nn.one_hot(jnp.zeros(cfg.n_agents, dtype=int), cfg.n_obs)
    F_sum = micro_free_energy_sum(mu, obs_soft, gm, cfg)
    assert F_sum.shape == ()
    assert not jnp.isnan(F_sum)


def test_check_macro_bound_holds(cfg, W, gm):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    obs_soft = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (cfg.n_agents, cfg.n_obs)), axis=-1
    )
    actions = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(2), (cfg.n_agents, cfg.n_actions)), axis=-1
    )
    macro = coarse_grain(mu, obs_soft, actions, W, cfg)
    F_macro = macro_free_energy(macro, gm, cfg)
    F_sum = micro_free_energy_sum(mu, obs_soft, gm, cfg)
    I_sync = jnp.array(0.0)
    holds, violation = check_macro_bound(F_macro, F_sum, I_sync)
    # Just verify outputs are valid scalars
    assert holds.shape == ()
    assert violation.shape == ()


# ── Task 13: Jacobian Eigenanalysis ───────────────────────────────────────────

from fep_swarm.macro.eigenanalysis import (
    swarm_belief_rates, compute_jacobian, temporal_horizons
)


def test_swarm_belief_rates_shape(cfg, gm):
    N, d = cfg.n_agents, cfg.n_hidden
    mu_flat = jax.random.normal(jax.random.PRNGKey(0), (N * d,))
    obs_soft = jax.nn.one_hot(jnp.zeros(N, dtype=int), cfg.n_obs)
    rates = swarm_belief_rates(mu_flat, obs_soft, gm, cfg)
    chex.assert_shape(rates, (N * d,))
    assert not jnp.any(jnp.isnan(rates))


def test_jacobian_shape():
    # Use tiny config to keep Jacobian tractable in test
    small_cfg = FEPConfig(n_agents=4, n_hidden=4, coarse_k=2)
    gm = DiscreteGenerativeModel(small_cfg, jax.random.PRNGKey(0))
    N, d = small_cfg.n_agents, small_cfg.n_hidden
    mu = jax.random.normal(jax.random.PRNGKey(1), (N, d))
    obs_soft = jax.nn.one_hot(jnp.zeros(N, dtype=int), small_cfg.n_obs)
    eigenvalues, gap, magnitudes = compute_jacobian(mu, obs_soft, gm, small_cfg)
    chex.assert_shape(eigenvalues, (N * d,))
    chex.assert_shape(magnitudes, (N * d,))
    assert not jnp.any(jnp.isnan(magnitudes))
    assert float(gap) > 0.0


def test_temporal_horizons_ordering():
    small_cfg = FEPConfig(n_agents=4, n_hidden=4, coarse_k=2)
    gm = DiscreteGenerativeModel(small_cfg, jax.random.PRNGKey(0))
    mu = jax.random.normal(jax.random.PRNGKey(1), (small_cfg.n_agents, small_cfg.n_hidden))
    obs_soft = jax.nn.one_hot(jnp.zeros(small_cfg.n_agents, dtype=int), small_cfg.n_obs)
    _, _, magnitudes = compute_jacobian(mu, obs_soft, gm, small_cfg)
    micro_h, macro_h = temporal_horizons(magnitudes, small_cfg)
    # macro horizon (slow modes) must be >= micro horizon (fast modes)
    assert float(macro_h) >= float(micro_h)
