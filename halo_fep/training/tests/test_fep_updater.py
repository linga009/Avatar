# halo_fep/training/tests/test_fep_updater.py
import jax
import jax.numpy as jnp
import numpy as np
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.memory.schema import Episode
from halo_fep.training.fep_updater import FEPUpdater


def make_episode(cfg):
    return Episode(
        query="q",
        tokens=np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32),
        swarm_mu=np.random.randn(cfg.n_agents, cfg.n_hidden).astype(np.float32),
        free_energy=1.0,
        free_energy_delta=-0.1,
    )


def test_update_returns_model():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    new_model = updater.update(model, carry, ep)
    assert new_model is not model


def test_update_changes_log_d():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    new_model = updater.update(model, carry, ep)
    assert not jnp.allclose(model.gm.log_D, new_model.gm.log_D)


def test_update_matrices_finite():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    new_model = updater.update(model, carry, ep)
    assert jnp.all(jnp.isfinite(new_model.gm.log_D))
    assert jnp.all(jnp.isfinite(new_model.gm.log_A))


def test_update_d_remains_distribution():
    """D property (softmax of log_D) must sum to ~1."""
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    for _ in range(5):
        model = updater.update(model, carry, ep)
    d_sum = float(jnp.sum(model.gm.D))
    assert abs(d_sum - 1.0) < 1e-4
