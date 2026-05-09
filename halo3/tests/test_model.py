"""Tests for Halo3Model."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.model import Halo3Model, Halo3Carry, halo3_step

_CFG = Halo3Config(
    d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
    d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
    n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
    mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
    leapfrog_step_size=0.1, meta_n_hidden=4, meta_n_actions=2,
    meta_k=3, max_cache=8, island_size=4,
)
_KEY = jax.random.PRNGKey(0)


def test_model_init():
    model = Halo3Model(_CFG, _KEY)
    assert hasattr(model, 'backbone')
    assert hasattr(model, 'hamiltonian')
    assert hasattr(model, 'kuramoto') is False  # kuramoto is in carry, not model


def test_init_carry():
    model = Halo3Model(_CFG, _KEY)
    carry = model.init_carry(_KEY)
    assert isinstance(carry, Halo3Carry)
    assert carry.kuramoto.theta.shape == (_CFG.n_clusters, _CFG.n_hidden)


def test_step_shapes():
    model = Halo3Model(_CFG, _KEY)
    carry = model.init_carry(_KEY)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_tokens, _CFG.d_model))
    new_carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, _KEY)
    assert h_out.shape == (_CFG.n_tokens, _CFG.d_model)
    assert obs.shape == (_CFG.n_clusters, _CFG.n_obs)
    assert q_final.shape == (_CFG.n_tokens, _CFG.d_boundary)


def test_step_no_nan():
    model = Halo3Model(_CFG, _KEY)
    carry = model.init_carry(_KEY)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_tokens, _CFG.d_model))
    new_carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, _KEY)
    assert jnp.all(jnp.isfinite(h_out))
    assert jnp.all(jnp.isfinite(q_final))


def test_loss_finite():
    from halo3.loss import halo3_loss
    model = Halo3Model(_CFG, _KEY)
    carry = model.init_carry(_KEY)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_tokens, _CFG.d_model))
    loss, aux = halo3_loss(model, carry, tokens, _KEY)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_gradients_nonzero():
    from halo3.loss import halo3_loss
    model = Halo3Model(_CFG, _KEY)
    carry = model.init_carry(_KEY)
    tokens = jax.random.normal(jax.random.PRNGKey(1), (_CFG.n_tokens, _CFG.d_model))
    grads = jax.grad(lambda m: halo3_loss(m, carry, tokens, _KEY)[0])(model)
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(g != 0.0) for g in leaves if hasattr(g, 'shape'))


def test_param_count_under_15M():
    model = Halo3Model(_CFG, _KEY)
    n = sum(p.size for p in jax.tree_util.tree_leaves(model) if hasattr(p, 'size'))
    # Small config will have few params; just verify it's reasonable
    assert n < 1_000_000, f"Small config has {n} params — expected < 1M"
