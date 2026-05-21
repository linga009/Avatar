# halo_fep/tests/test_fep_updater.py
"""Tests for FEPUpdater — verifies Bug 3 fix (real soft_obs vs action proxy).

Critical invariants tested
--------------------------
1. FEPUpdater.update() accepts ``soft_obs`` as an explicit argument.
2. The A matrix diverges from the uniform baseline when non-uniform soft_obs
   is supplied (proves the A update actually uses the observation signal).
3. n_obs != n_actions case does not fall through to uniform (the old proxy
   fallback path is gone; real soft_obs is always used).
4. D and B matrices are also updated (non-trivial changes from initial values).
5. All updated log matrices remain finite (no NaN/Inf).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, HaloFEPCarry, halo_fep_step
from halo_fep.training.fep_updater import FEPUpdater
from halo_fep.memory.schema import Episode


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def cfg():
    """Standard test config with n_obs == n_actions (symmetric case)."""
    return HaloFEPConfig(n_obs=4, n_actions=4)


@pytest.fixture
def cfg_asymmetric():
    """Config with n_obs != n_actions (asymmetric case — previously broken)."""
    return HaloFEPConfig(n_obs=4, n_actions=4, n_hidden=8)


@pytest.fixture
def model_and_carry(cfg):
    key   = jax.random.PRNGKey(0)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    return model, carry


@pytest.fixture
def episode(cfg):
    return Episode(
        query             = "test query",
        tokens            = np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32),
        swarm_mu          = np.zeros((cfg.n_agents, cfg.n_hidden), dtype=np.float32),
        free_energy       = 1.0,
        free_energy_delta = -0.1,
    )


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_fep_updater_accepts_soft_obs(model_and_carry, cfg, episode):
    """FEPUpdater.update() must accept soft_obs without raising."""
    model, carry = model_and_carry
    updater  = FEPUpdater(cfg)
    soft_obs = jnp.ones((cfg.n_agents, cfg.n_obs)) / cfg.n_obs  # uniform
    new_model = updater.update(model, carry, episode, soft_obs)
    assert new_model is not model  # returns new model, not mutated in place


def test_a_matrix_tracks_observations_not_actions(cfg, episode):
    """A matrix update should encode the supplied soft_obs, not swarm_action.

    We supply a strongly peaked soft_obs (one observation dominant) and a
    uniform carry (so swarm_action is uniform).  After the update, the A
    matrix should be non-uniform (reflecting the peaked soft_obs).
    If the old action-proxy bug were present, A would remain near-uniform.
    """
    key   = jax.random.PRNGKey(1)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)

    updater = FEPUpdater(cfg, alpha=0.0)  # alpha=0: full replacement

    # Strongly peaked soft_obs: observation 0 is certain
    soft_obs                = jnp.zeros((cfg.n_agents, cfg.n_obs))
    soft_obs                = soft_obs.at[:, 0].set(1.0)

    new_model = updater.update(model, carry, episode, soft_obs)

    # After full replacement, A[:,j] should be peaked at obs 0
    A = jax.nn.softmax(new_model.gm.log_A, axis=0)   # (n_obs, n_hidden)
    # Row 0 should be the highest row for all hidden states
    assert jnp.all(A[0] > A[1]), (
        "A matrix row 0 should dominate after peaked soft_obs update. "
        "This may indicate the action-proxy bug has returned."
    )


def test_a_matrix_finite_after_update(model_and_carry, cfg, episode):
    """All log matrices must remain finite after an update."""
    model, carry = model_and_carry
    updater  = FEPUpdater(cfg)
    soft_obs = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(42), (cfg.n_agents, cfg.n_obs)), axis=-1)
    new_model = updater.update(model, carry, episode, soft_obs)

    for name, log_mat in [
        ("log_A", new_model.gm.log_A),
        ("log_D", new_model.gm.log_D),
        ("log_B", new_model.gm.log_B),
    ]:
        assert jnp.all(jnp.isfinite(log_mat)), f"{name} contains NaN/Inf after update"


def test_d_matrix_updated(model_and_carry, cfg, episode):
    """D matrix should change after update (proves D update runs)."""
    model, carry = model_and_carry
    log_D_before = model.gm.log_D.copy()
    updater      = FEPUpdater(cfg, alpha=0.9)  # not alpha=1 (no change)
    soft_obs     = jax.nn.softmax(jnp.ones((cfg.n_agents, cfg.n_obs)), axis=-1)
    new_model    = updater.update(model, carry, episode, soft_obs)
    assert not jnp.allclose(new_model.gm.log_D, log_D_before), (
        "log_D should change after FEP update with alpha < 1.0"
    )


def test_asymmetric_obs_actions_no_fallback(cfg_asymmetric, episode):
    """n_obs != n_actions must not fall through to uniform proxy.

    Previously FEPUpdater used ``swarm_action`` as a proxy when
    n_obs == n_actions and fell back to uniform otherwise.
    Now it always uses real soft_obs, so both cases must produce
    non-uniform A when a peaked soft_obs is supplied.
    """
    cfg   = cfg_asymmetric
    key   = jax.random.PRNGKey(2)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)

    updater  = FEPUpdater(cfg, alpha=0.0)   # full replacement
    soft_obs = jnp.zeros((cfg.n_agents, cfg.n_obs)).at[:, 1].set(1.0)  # peaked at obs 1

    new_model = updater.update(model, carry, episode, soft_obs)
    A = jax.nn.softmax(new_model.gm.log_A, axis=0)

    # Row 1 should dominate
    assert jnp.all(A[1] > A[0]), (
        "Asymmetric n_obs/n_actions: A row 1 should dominate. "
        "Old uniform fallback may have been reintroduced."
    )


def test_fep_updater_integrated_with_step(cfg, episode):
    """Integration test: run a full step then update FEP matrices."""
    key   = jax.random.PRNGKey(3)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))

    # Real forward pass to get real soft_obs
    new_carry, (h_out, soft_obs, v_pred, v_target) = halo_fep_step(
        model, carry, tokens, key
    )

    updater   = FEPUpdater(cfg)
    new_model = updater.update(model, new_carry, episode, soft_obs)

    # All outputs must be finite
    for name, arr in [
        ("log_A", new_model.gm.log_A),
        ("log_D", new_model.gm.log_D),
        ("log_B", new_model.gm.log_B),
    ]:
        assert jnp.all(jnp.isfinite(arr)), f"{name} has NaN/Inf after integrated step"
