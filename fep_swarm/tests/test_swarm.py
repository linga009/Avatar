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
