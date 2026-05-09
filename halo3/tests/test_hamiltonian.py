"""Tests for Hamiltonian Neural ODE."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.hamiltonian import LearnedHamiltonian, v_ads, leapfrog_step, leapfrog_integrate, MomentumInitializer

_CFG = Halo3Config(d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8, d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4, n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4, mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2, leapfrog_step_size=0.1, meta_n_hidden=4, meta_n_actions=2, meta_k=3, max_cache=8, island_size=4)
_KEY = jax.random.PRNGKey(0)

def test_v_ads_scalar():
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    assert v_ads(q, _CFG.init_curvature).shape == ()

def test_v_ads_finite():
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    assert jnp.isfinite(v_ads(q, _CFG.init_curvature))

def test_hamiltonian_scalar():
    H = LearnedHamiltonian(_CFG, _KEY)
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    p = jax.random.normal(jax.random.PRNGKey(1), q.shape)
    e = H(q, p)
    assert e.shape == ()
    assert jnp.isfinite(e)

def test_leapfrog_energy_conservation():
    H = LearnedHamiltonian(_CFG, _KEY)
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    p = jax.random.normal(jax.random.PRNGKey(1), q.shape)
    E0 = H(q, p)
    q_f, p_f = leapfrog_integrate(H, q, p, _CFG.n_leapfrog_steps, _CFG.leapfrog_step_size)
    Ef = H(q_f, p_f)
    assert jnp.abs(Ef - E0) / (jnp.abs(E0) + 1e-8) < 0.1

def test_leapfrog_shapes():
    H = LearnedHamiltonian(_CFG, _KEY)
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    p = jax.random.normal(jax.random.PRNGKey(1), q.shape)
    q_f, p_f = leapfrog_integrate(H, q, p, 2, 0.1)
    assert q_f.shape == q.shape
    assert p_f.shape == p.shape

def test_leapfrog_no_nan():
    H = LearnedHamiltonian(_CFG, _KEY)
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    p = jax.random.normal(jax.random.PRNGKey(1), q.shape)
    q_f, p_f = leapfrog_integrate(H, q, p, 2, 0.1)
    assert jnp.all(jnp.isfinite(q_f))
    assert jnp.all(jnp.isfinite(p_f))

def test_momentum_init_shape():
    mi = MomentumInitializer(_CFG, _KEY)
    h = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_model))
    assert mi(h).shape == (_CFG.n_tokens, _CFG.d_boundary)

def test_gradients_through_leapfrog():
    H = LearnedHamiltonian(_CFG, _KEY)
    q = jax.random.normal(_KEY, (_CFG.n_tokens, _CFG.d_boundary))
    p = jax.random.normal(jax.random.PRNGKey(1), q.shape)
    grads = jax.grad(lambda h: jnp.sum(leapfrog_integrate(h, q, p, 2, 0.1)[0]))(H)
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0.0) for g in leaves if hasattr(g, 'shape'))
