import jax
import jax.numpy as jnp
import pytest
import inspect
import chex
from fep_swarm.config import FEPConfig
from fep_swarm.agent.markov_blanket import AgentState, check_blanket_independence


@pytest.fixture
def cfg():
    return FEPConfig()


def test_agent_state_fields(cfg):
    mu = jnp.zeros(cfg.n_hidden)
    action = jnp.zeros(cfg.n_actions)
    obs = jnp.zeros(cfg.n_obs)
    state = AgentState(mu=mu, action=action, obs=obs)
    chex.assert_shape(state.mu, (cfg.n_hidden,))
    chex.assert_shape(state.action, (cfg.n_actions,))
    chex.assert_shape(state.obs, (cfg.n_obs,))


def test_blanket_independence_low_for_uncorrelated(cfg):
    key = jax.random.PRNGKey(0)
    N = 100
    mu = jax.random.normal(key, (N, cfg.n_hidden))
    eta = jax.random.normal(jax.random.PRNGKey(1), (N, cfg.n_hidden))
    obs = jax.random.normal(jax.random.PRNGKey(2), (N, cfg.n_obs))
    # When mu and eta are independent given obs, MI proxy should be low
    mi_proxy = check_blanket_independence(mu, eta, obs)
    assert mi_proxy < 1.0  # near 0 for independent random variables
