import jax
import jax.numpy as jnp
import pytest
import chex
from fep_swarm.config import FEPConfig
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel


@pytest.fixture
def cfg():
    return FEPConfig()


@pytest.fixture
def gm(cfg):
    return DiscreteGenerativeModel(cfg, jax.random.PRNGKey(0))


def test_A_shape(gm, cfg):
    chex.assert_shape(gm.A, (cfg.n_obs, cfg.n_hidden))


def test_A_column_stochastic(gm, cfg):
    col_sums = gm.A.sum(axis=0)
    assert jnp.allclose(col_sums, jnp.ones(cfg.n_hidden), atol=1e-5)


def test_B_shape(gm, cfg):
    chex.assert_shape(gm.B, (cfg.n_hidden, cfg.n_hidden, cfg.n_actions))


def test_B_column_stochastic(gm, cfg):
    for a in range(cfg.n_actions):
        col_sums = gm.B[:, :, a].sum(axis=0)
        assert jnp.allclose(col_sums, jnp.ones(cfg.n_hidden), atol=1e-5)


def test_D_sums_to_one(gm, cfg):
    assert jnp.allclose(gm.D.sum(), 1.0, atol=1e-5)


def test_C_shape(gm, cfg):
    chex.assert_shape(gm.C, (cfg.n_obs,))


def test_no_nan(gm):
    assert not jnp.any(jnp.isnan(gm.A))
    assert not jnp.any(jnp.isnan(gm.B))
    assert not jnp.any(jnp.isnan(gm.D))


from fep_swarm.generative_model.continuous_likelihood import ContinuousLikelihood


def test_ode_output_shape(cfg):
    cl = ContinuousLikelihood(cfg, jax.random.PRNGKey(1))
    x0 = jnp.zeros(cfg.obs_dim)
    out = cl(eta_idx=0, x0=x0)
    chex.assert_shape(out, (cfg.obs_dim,))


def test_ode_no_nan(cfg):
    cl = ContinuousLikelihood(cfg, jax.random.PRNGKey(2))
    x0 = jax.random.normal(jax.random.PRNGKey(3), (cfg.obs_dim,))
    out = cl(eta_idx=2, x0=x0)
    assert not jnp.any(jnp.isnan(out))


def test_ode_different_eta_gives_different_output(cfg):
    cl = ContinuousLikelihood(cfg, jax.random.PRNGKey(4))
    x0 = jnp.zeros(cfg.obs_dim)
    out0 = cl(eta_idx=0, x0=x0)
    out1 = cl(eta_idx=1, x0=x0)
    assert not jnp.allclose(out0, out1)
