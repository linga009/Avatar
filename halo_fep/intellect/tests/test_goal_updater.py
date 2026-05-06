# halo_fep/intellect/tests/test_goal_updater.py
import jax
import numpy as np
from unittest.mock import MagicMock, patch
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.intellect.goal_updater import GoalUpdater


def make_model(cfg):
    return HaloFEPModel(cfg, jax.random.PRNGKey(0))


def make_mock_embedder(n_obs=4, d_model=256):
    emb = MagicMock()
    emb.embed_text.return_value = np.random.randn(384).astype(np.float32)
    return emb


def test_update_goal_returns_model():
    cfg = HaloFEPConfig()
    model = make_model(cfg)
    updater = GoalUpdater(cfg)
    updater._embedder = make_mock_embedder(cfg.n_obs)
    new_model = updater.update_goal(model, "understand consciousness")
    assert new_model is not model  # new tree


def test_update_goal_changes_log_c():
    cfg = HaloFEPConfig()
    model = make_model(cfg)
    orig_log_c = np.array(model.gm.log_C)
    updater = GoalUpdater(cfg)
    updater._embedder = make_mock_embedder(cfg.n_obs)
    new_model = updater.update_goal(model, "something specific")
    new_log_c = np.array(new_model.gm.log_C)
    assert not np.allclose(orig_log_c, new_log_c)


def test_decay_toward_uniform():
    cfg = HaloFEPConfig()
    model = make_model(cfg)
    updater = GoalUpdater(cfg)
    new_model = updater.decay(model)
    # After decay, log_C should still be valid (no NaN/Inf)
    import jax.numpy as jnp
    assert jnp.all(jnp.isfinite(new_model.gm.log_C))
