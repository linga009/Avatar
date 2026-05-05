import jax
import jax.numpy as jnp
import pytest
import chex
from fep_swarm.config import FEPConfig
from fep_swarm.swarm.environment import init_env, observe, step_env
from fep_swarm.data.synthetic_world import make_tmaze


@pytest.fixture
def cfg():
    return FEPConfig(n_agents=16)  # small N for test speed


@pytest.fixture
def tmaze(cfg):
    return make_tmaze(cfg)


def test_init_env_shape(cfg):
    env = init_env(cfg, jax.random.PRNGKey(0))
    chex.assert_shape(env.eta, (cfg.n_hidden,))
    assert int(env.step) == 0


def test_observe_shape(cfg, tmaze):
    A, _, _, _ = tmaze
    env = init_env(cfg, jax.random.PRNGKey(0))
    obs = observe(env, cfg, A, jax.random.PRNGKey(1))
    chex.assert_shape(obs, (cfg.n_agents,))
    assert jnp.all(obs >= 0) and jnp.all(obs < cfg.n_obs)


def test_step_env_increments_step(cfg, tmaze):
    A, B, _, _ = tmaze
    env = init_env(cfg, jax.random.PRNGKey(0))
    actions = jnp.zeros((cfg.n_agents, cfg.n_actions)).at[:, 0].set(1.0)
    env2 = step_env(env, actions, B, cfg, jax.random.PRNGKey(2))
    assert int(env2.step) == 1


from fep_swarm.swarm.coupling import build_coupling_matrix, apply_coupling


def test_all2all_coupling_matrix(cfg):
    W = build_coupling_matrix(cfg, jax.random.PRNGKey(0))
    chex.assert_shape(W, (cfg.n_agents, cfg.n_agents))
    # All rows sum to 1
    assert jnp.allclose(W.sum(axis=1), jnp.ones(cfg.n_agents), atol=1e-5)


def test_sparse_coupling_matrix():
    cfg = FEPConfig(n_agents=16, topology="sparse", sparse_p=0.5)
    W = build_coupling_matrix(cfg, jax.random.PRNGKey(0))
    chex.assert_shape(W, (cfg.n_agents, cfg.n_agents))
    # No self-loops
    assert jnp.all(jnp.diag(W) == 0.0)


def test_apply_coupling_shape(cfg):
    W = build_coupling_matrix(cfg, jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_agents, cfg.n_obs))
    obs = jax.nn.softmax(obs, axis=-1)
    actions = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(2), (cfg.n_agents, cfg.n_actions)),
        axis=-1
    )
    obs_new = apply_coupling(obs, actions, W, cfg)
    chex.assert_shape(obs_new, (cfg.n_agents, cfg.n_obs))


def test_kappa_zero_no_influence(cfg):
    cfg_zero = FEPConfig(n_agents=16, kappa=0.0)
    W = build_coupling_matrix(cfg_zero, jax.random.PRNGKey(0))
    obs = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(1), (cfg_zero.n_agents, cfg_zero.n_obs)),
        axis=-1
    )
    actions = jax.nn.softmax(
        jax.random.normal(jax.random.PRNGKey(2), (cfg_zero.n_agents, cfg_zero.n_actions)),
        axis=-1
    )
    obs_new = apply_coupling(obs, actions, W, cfg_zero)
    assert jnp.allclose(obs_new, obs, atol=1e-6)


from fep_swarm.swarm.synchrony import synchrony_metric, mutual_information_estimate


def test_synchrony_metric_zero_for_identical(cfg):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    S = synchrony_metric(mu, mu)  # mu_dot = 0 everywhere
    assert float(S) == pytest.approx(0.0, abs=1e-6)


def test_synchrony_metric_positive_for_different(cfg):
    mu = jax.random.normal(jax.random.PRNGKey(0), (cfg.n_agents, cfg.n_hidden))
    mu_prev = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_agents, cfg.n_hidden))
    S = synchrony_metric(mu, mu_prev)
    assert float(S) > 0.0


def test_mi_estimate_positive(cfg):
    T = 50
    mu_history = jax.random.normal(
        jax.random.PRNGKey(0), (T, cfg.n_agents, cfg.n_hidden)
    )
    mi = mutual_information_estimate(mu_history)
    assert float(mi) >= 0.0
