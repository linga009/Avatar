"""Tests for Bohmian Kuramoto oscillators on n-torus."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.kuramoto import (
    KuramotoState, init_kuramoto, kuramoto_step, kuramoto_action,
    order_parameter, quantum_potential, dual_order_parameters,
)

_CFG = Halo3Config(d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8, d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4, n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4, mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2, meta_n_hidden=4, meta_n_actions=2, meta_k=3, max_cache=8, island_size=4)
_KEY = jax.random.PRNGKey(0)


def test_init_shapes():
    state = init_kuramoto(_CFG, _KEY)
    assert state.theta.shape == (_CFG.n_clusters, _CFG.n_hidden)
    assert state.omega.shape == (_CFG.n_clusters, _CFG.n_hidden)


def test_phases_in_range():
    state = init_kuramoto(_CFG, _KEY)
    assert jnp.all(state.theta >= 0.0)
    assert jnp.all(state.theta < 2 * jnp.pi)


def test_step_updates_theta():
    state = init_kuramoto(_CFG, _KEY)
    obs = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_clusters, _CFG.n_obs))
    new_state = kuramoto_step(state, obs, _CFG)
    assert not jnp.allclose(new_state.theta, state.theta)


def test_step_keeps_phases_in_range():
    state = init_kuramoto(_CFG, _KEY)
    obs = jax.random.normal(_KEY, (_CFG.n_clusters, _CFG.n_obs))
    for _ in range(100):
        state = kuramoto_step(state, obs, _CFG)
    assert jnp.all(state.theta >= 0.0)
    assert jnp.all(state.theta < 2 * jnp.pi)


def test_action_valid_probs():
    state = init_kuramoto(_CFG, _KEY)
    actions = kuramoto_action(state, _CFG.n_actions)
    assert actions.shape == (_CFG.n_clusters, _CFG.n_actions)
    assert jnp.allclose(jnp.sum(actions, axis=-1), 1.0, atol=1e-5)
    assert jnp.all(actions >= 0.0)


def test_order_parameter_range():
    state = init_kuramoto(_CFG, _KEY)
    r = order_parameter(state.theta)
    assert r.shape == (_CFG.n_hidden,)
    assert jnp.all(r >= 0.0)
    assert jnp.all(r <= 1.0 + 1e-5)


def test_synchronized_high_order():
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    r = order_parameter(theta)
    assert jnp.all(r > 0.99)


def test_no_nan():
    state = init_kuramoto(_CFG, _KEY)
    obs = jax.random.normal(_KEY, (_CFG.n_clusters, _CFG.n_obs))
    new_state = kuramoto_step(state, obs, _CFG)
    assert jnp.all(jnp.isfinite(new_state.theta))
    assert jnp.all(jnp.isfinite(kuramoto_action(new_state, _CFG.n_actions)))


# --- Bohmian extensions ---

def test_quantum_potential_shape():
    state = init_kuramoto(_CFG, _KEY)
    Q = quantum_potential(state.theta)
    assert Q.shape == (_CFG.n_clusters, _CFG.n_hidden)


def test_quantum_potential_finite():
    state = init_kuramoto(_CFG, _KEY)
    Q = quantum_potential(state.theta)
    assert jnp.all(jnp.isfinite(Q))


def test_quantum_potential_zero_when_synchronized():
    """All phases equal → Q should be zero (no anti-bunching needed)."""
    theta = jnp.ones((_CFG.n_clusters, _CFG.n_hidden)) * jnp.pi
    Q = quantum_potential(theta)
    assert jnp.allclose(Q, 0.0, atol=1e-6)


def test_quantum_potential_nonzero_when_spread():
    """Diverse phases → Q should be nonzero."""
    theta = jax.random.uniform(_KEY, (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    Q = quantum_potential(theta)
    assert jnp.any(jnp.abs(Q) > 1e-4)


def test_pilot_wave_step():
    """Step with explicit pilot wave should produce different result than without."""
    state = init_kuramoto(_CFG, _KEY)
    obs = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_clusters, _CFG.n_obs))
    pilot = jax.random.normal(jax.random.PRNGKey(2), (_CFG.n_clusters, _CFG.d_boundary))

    state_classical = kuramoto_step(state, obs, _CFG, pilot_wave=None)
    state_bohmian = kuramoto_step(state, obs, _CFG, pilot_wave=pilot)

    assert not jnp.allclose(state_classical.theta, state_bohmian.theta)


def test_pilot_wave_no_nan():
    state = init_kuramoto(_CFG, _KEY)
    obs = jax.random.normal(_KEY, (_CFG.n_clusters, _CFG.n_obs))
    pilot = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_clusters, _CFG.d_boundary))
    new_state = kuramoto_step(state, obs, _CFG, pilot_wave=pilot)
    assert jnp.all(jnp.isfinite(new_state.theta))


# --- Dual-process body tension ---

def test_dual_order_parameters_zero_tension():
    """Both populations fully synchronized → tension ≈ 0."""
    # All phases = 0: both analytical and creative halves are perfectly synced
    theta = jnp.zeros((_CFG.n_clusters, _CFG.n_hidden))
    r_a, r_c, tension = dual_order_parameters(theta)
    assert float(r_a) > 0.99
    assert float(r_c) > 0.99
    assert float(tension) < 0.05


def test_dual_order_parameters_high_tension():
    """Analytical synced, creative spread → tension ≈ 1."""
    mid = _CFG.n_hidden // 2
    # Analytical half: all phases = 0 (fully synced, r_a → 1)
    theta_a = jnp.zeros((_CFG.n_clusters, mid))
    # Creative half: uniformly spread around circle (r_c → 0)
    angles = jnp.linspace(0, 2 * jnp.pi, _CFG.n_clusters, endpoint=False)
    theta_c = jnp.tile(angles[:, None], (1, mid))
    theta = jnp.concatenate([theta_a, theta_c], axis=1)
    r_a, r_c, tension = dual_order_parameters(theta)
    assert float(r_a) > 0.95
    assert float(r_c) < 0.1
    assert float(tension) > 0.85


def test_dual_order_parameters_range():
    """Tension is always in [0, 1] on random phases."""
    theta = jax.random.uniform(_KEY, (_CFG.n_clusters, _CFG.n_hidden)) * 2 * jnp.pi
    r_a, r_c, tension = dual_order_parameters(theta)
    assert 0.0 <= float(r_a) <= 1.0 + 1e-5
    assert 0.0 <= float(r_c) <= 1.0 + 1e-5
    assert 0.0 <= float(tension) <= 1.0 + 1e-5
