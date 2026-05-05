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


from fep_swarm.agent.belief_update import free_energy, belief_update
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel


@pytest.fixture
def gm(cfg):
    return DiscreteGenerativeModel(cfg, jax.random.PRNGKey(10))


def test_free_energy_scalar(cfg, gm):
    mu = jnp.zeros(cfg.n_hidden)
    F = free_energy(mu, obs_idx=0, gm=gm)
    assert F.shape == ()
    assert not jnp.isnan(F)


def test_free_energy_decreases_over_steps(cfg, gm):
    mu = jax.random.normal(jax.random.PRNGKey(5), (cfg.n_hidden,))
    obs_idx = 0
    F_init = free_energy(mu, obs_idx, gm)
    mu_updated = belief_update(mu, obs_idx, gm, cfg)
    F_final = free_energy(mu_updated, obs_idx, gm)
    assert F_final < F_init


def test_belief_update_no_nan(cfg, gm):
    mu = jnp.zeros(cfg.n_hidden)
    mu_out = belief_update(mu, obs_idx=1, gm=gm, cfg=cfg)
    assert not jnp.any(jnp.isnan(mu_out))
    chex.assert_shape(mu_out, (cfg.n_hidden,))


def test_belief_update_no_eta_dependency():
    """Structural test: belief_update signature must not include eta."""
    import inspect
    sig = inspect.signature(belief_update)
    assert "eta" not in sig.parameters
