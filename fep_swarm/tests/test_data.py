import jax
import jax.numpy as jnp
import pytest
import chex
from fep_swarm.config import FEPConfig
from fep_swarm.data.synthetic_world import make_tmaze, sample_obs, transition_world


@pytest.fixture
def cfg():
    return FEPConfig()


def test_tmaze_shapes(cfg):
    A, B, C, D = make_tmaze(cfg)
    chex.assert_shape(A, (cfg.n_obs, cfg.n_hidden))
    chex.assert_shape(B, (cfg.n_hidden, cfg.n_hidden, cfg.n_actions))
    chex.assert_shape(C, (cfg.n_obs,))
    chex.assert_shape(D, (cfg.n_hidden,))


def test_tmaze_A_column_stochastic(cfg):
    A, _, _, _ = make_tmaze(cfg)
    assert jnp.allclose(A.sum(axis=0), jnp.ones(cfg.n_hidden), atol=1e-5)


def test_sample_obs_valid(cfg):
    A, _, _, D = make_tmaze(cfg)
    eta = D  # start state
    obs = sample_obs(eta, A, jax.random.PRNGKey(0))
    assert 0 <= int(obs) < cfg.n_obs


def test_transition_world_valid(cfg):
    A, B, C, D = make_tmaze(cfg)
    action = jax.nn.one_hot(1, cfg.n_actions)  # go left
    eta_new = transition_world(D, action, B, jax.random.PRNGKey(0))
    chex.assert_shape(eta_new, (cfg.n_hidden,))
    assert jnp.allclose(eta_new.sum(), 1.0, atol=1e-5)
