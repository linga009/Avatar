# halo_fep/training/tests/test_hyperbolic_pretrain.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.training.hyperbolic_pretrain import (
    poincare_distance,
    poincare_loss,
    run_hyperbolic_pretrain,
)


def test_poincare_distance_is_zero_for_same_point():
    u = jnp.array([0.1, 0.2])
    d = poincare_distance(u, u)
    assert float(d) < 1e-4


def test_poincare_distance_increases_near_boundary():
    """Points near the disk boundary are far from the center."""
    center = jnp.zeros(2)
    near_boundary = jnp.array([0.99, 0.0])
    d_near = poincare_distance(center, near_boundary)
    d_mid  = poincare_distance(center, jnp.array([0.5, 0.0]))
    assert float(d_near) > float(d_mid)


def test_poincare_distance_is_symmetric():
    u = jnp.array([0.3, 0.1])
    v = jnp.array([-0.2, 0.4])
    assert jnp.allclose(poincare_distance(u, v), poincare_distance(v, u), atol=1e-5)


def test_poincare_loss_is_scalar():
    dim = 4
    n_entities = 10
    embeddings = jnp.array(
        np.random.uniform(-0.5, 0.5, (n_entities, dim)).astype(np.float32)
    )
    neg_idxs = jnp.array([2, 3, 4])
    loss = poincare_loss(embeddings, u_idx=0, v_idx=1, neg_idxs=neg_idxs)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_run_hyperbolic_pretrain_returns_model():
    """run_hyperbolic_pretrain should return a HaloFEPModel (may mock dataset)."""
    from unittest.mock import patch
    cfg = HaloFEPConfig(n_tokens=32)
    key = jax.random.PRNGKey(0)
    model = HaloFEPModel(cfg, key)

    # Fake triples: list of dicts with head/relation/tail indices
    fake_triples = [{"head": i % 5, "relation": 0, "tail": (i + 1) % 5}
                    for i in range(20)]

    with patch("halo_fep.training.hyperbolic_pretrain.load_dataset",
               return_value=fake_triples):
        updated_model = run_hyperbolic_pretrain(
            model, cfg, key=key, n_steps=5, n_entities=5
        )

    # Model should be returned (backbone updated)
    assert updated_model is not None
