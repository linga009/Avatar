# halo_fep/training/tests/test_mesu.py
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pytest
from halo_fep.training.mesu import mesu


def test_mesu_returns_gradient_transformation():
    opt = mesu(lr=1e-3, eta=0.01)
    assert hasattr(opt, "init")
    assert hasattr(opt, "update")


def test_mesu_init_state_has_sigma():
    opt = mesu(lr=1e-3, eta=0.01)
    params = {"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}
    state = opt.init(params)
    assert "sigma" in state
    # sigma should be all-ones initially (uninformative prior)
    for leaf in jtu.tree_leaves(state["sigma"]):
        assert jnp.allclose(leaf, jnp.ones_like(leaf))


def test_mesu_updates_reduce_shape_matches_params():
    opt = mesu(lr=1e-3, eta=0.01)
    params = {"w": jnp.ones((3, 3))}
    state = opt.init(params)
    grads = {"w": jnp.ones((3, 3)) * 0.1}
    updates, new_state = opt.update(grads, state)
    assert updates["w"].shape == (3, 3)
    assert new_state["sigma"]["w"].shape == (3, 3)


def test_mesu_sigma_increases_with_large_gradients():
    """High gradient variance should increase uncertainty sigma."""
    opt = mesu(lr=1e-4, eta=0.5)
    params = {"w": jnp.zeros(4)}
    state = opt.init(params)
    # Large gradient
    grads = {"w": jnp.ones(4) * 10.0}
    _, state2 = opt.update(grads, state)
    # sigma should have grown from 1.0 toward 10^2 = 100
    assert jnp.all(state2["sigma"]["w"] > state["sigma"]["w"])


def test_mesu_update_scales_by_inverse_sigma():
    """Updates should be smaller when sigma is large."""
    opt = mesu(lr=1.0, eta=0.0, epsilon=0.0)  # no sigma adaptation
    params = {"w": jnp.zeros(1)}

    # Low sigma (uncertain parameter) -> large update
    state_low = {"sigma": {"w": jnp.ones(1) * 0.1}}
    grads = {"w": jnp.ones(1)}
    updates_low, _ = opt.update(grads, state_low)

    # High sigma (certain parameter) -> small update
    state_high = {"sigma": {"w": jnp.ones(1) * 10.0}}
    updates_high, _ = opt.update(grads, state_high)

    assert abs(updates_low["w"][0]) > abs(updates_high["w"][0])
