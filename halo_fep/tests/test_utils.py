import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.utils import compute_free_energy


def test_compute_free_energy_scalar():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    fe = compute_free_energy(carry, model)
    assert fe.shape == ()


def test_compute_free_energy_nonnegative():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    fe = compute_free_energy(carry, model)
    assert float(fe) >= 0.0


def test_config_has_wake_threshold():
    cfg = HaloFEPConfig()
    assert cfg.wake_threshold == 2.5


def test_config_has_tick_interval():
    cfg = HaloFEPConfig()
    assert cfg.tick_interval == 60


def test_config_wake_threshold_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        HaloFEPConfig(wake_threshold=-1.0)


def test_config_tick_interval_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        HaloFEPConfig(tick_interval=0)
