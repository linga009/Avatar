"""Integration test: run 2 heartbeat ticks with all external calls mocked."""
import jax
import numpy as np
import jax.numpy as jnp
from unittest.mock import MagicMock, patch
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.main import HeartbeatLoop


def make_mock_perception(cfg):
    p = MagicMock()
    p.embed.return_value = jnp.zeros((cfg.n_tokens, cfg.d_model))
    p.embed_query.return_value = np.zeros(cfg.d_model, dtype=np.float32)
    p.query_from_beliefs.return_value = "test query"
    return p


def make_mock_memory():
    m = MagicMock()
    m.retrieve.return_value = []
    m.get_high_confidence.return_value = []
    return m


def test_heartbeat_tick_runs():
    cfg   = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    loop  = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=make_mock_perception(cfg),
        memory=make_mock_memory(),
    )
    loop.tick()   # should not raise


def test_heartbeat_two_ticks():
    cfg   = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    loop  = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=make_mock_perception(cfg),
        memory=make_mock_memory(),
    )
    loop.tick()
    loop.tick()


def test_heartbeat_perception_failure_continues():
    """If perception fails, tick should log and return gracefully."""
    cfg   = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    bad_perception = MagicMock()
    bad_perception.query_from_beliefs.return_value = "q"
    bad_perception.embed.side_effect = Exception("network error")
    loop  = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=bad_perception,
        memory=make_mock_memory(),
    )
    loop.tick()  # should not raise
