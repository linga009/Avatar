import jax
import tempfile
import os
from halo_fep.config import HaloFEPConfig
from halo_fep.training.bootstrap import run_bootstrap, save_checkpoint, load_checkpoint


def test_save_load_roundtrip():
    cfg = HaloFEPConfig(n_tokens=32)
    from halo_fep.model import HaloFEPModel
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt")
        save_checkpoint(model, path)
        loaded = load_checkpoint(cfg, path)
    # Check a weight is numerically identical
    import jax.numpy as jnp
    assert jnp.allclose(model.gm.log_D, loaded.gm.log_D)


def test_run_bootstrap_minimal():
    """Run 2 steps (not 5000) to verify the loop executes without error."""
    cfg = HaloFEPConfig(n_tokens=32)
    with tempfile.TemporaryDirectory() as d:
        model = run_bootstrap(cfg, n_pretrain_steps=2, n_synthetic_episodes=2,
                              checkpoint_dir=os.path.join(d, "ckpt"), seed=0)
    assert model is not None


import jax.numpy as jnp
from halo_fep.model import HaloFEPModel
from halo_fep.training.bootstrap import _multiscale_elbo_loss


def test_multiscale_loss_is_scalar():
    cfg = HaloFEPConfig(n_tokens=32)
    key = jax.random.PRNGKey(0)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    tokens = jnp.zeros((32, cfg.d_model))
    loss, aux = _multiscale_elbo_loss(model, carry, tokens, key, strides=(1, 4))
    assert loss.shape == ()
    assert float(loss) >= 0.0


def test_multiscale_loss_differs_from_single_scale():
    """Multi-scale loss should differ from single-stride loss."""
    cfg = HaloFEPConfig(n_tokens=32)
    key = jax.random.PRNGKey(1)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    tokens = jax.random.normal(key, (32, cfg.d_model))
    from halo_fep.loss import unified_elbo_loss
    single, _ = unified_elbo_loss(model, carry, tokens, key)
    multi, _  = _multiscale_elbo_loss(model, carry, tokens, key, strides=(1, 4))
    # They should not be identical (strides produce different subsampled inputs)
    assert not jnp.allclose(single, multi, atol=1e-5)
