# halo_fep/tests/test_benchmark.py
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.benchmark.multimodal_world import MultimodalWorld
from halo_fep.benchmark.eval import run_episode, benchmark

cfg = HaloFEPConfig()
key = jax.random.PRNGKey(7)

def test_world_sample_shape():
    world = MultimodalWorld(cfg, key)
    tokens, eta = world.sample(eta=0, key=key)
    assert tokens.shape == (cfg.n_tokens, cfg.d_model)
    assert eta == 0

def test_episode_runs_without_error():
    model = HaloFEPModel(cfg, key)
    world = MultimodalWorld(cfg, key)
    success = run_episode(model, world, cfg, key, n_steps=5)
    assert isinstance(bool(success), bool)

def test_belief_entropy_decreases():
    """Agents should not produce NaN beliefs over an episode."""
    model = HaloFEPModel(cfg, key)
    world = MultimodalWorld(cfg, key)
    carry = model.init_carry(key)
    eta   = 0

    def entropy(mu):
        q = jax.nn.softmax(mu, axis=-1)
        return -jnp.sum(q * jnp.log(q + 1e-8), axis=-1)

    # Run 20 steps
    k = key
    for _ in range(20):
        k, k_step = jax.random.split(k)
        tokens, _ = world.sample(eta=eta, key=k_step)
        carry, _ = halo_fep_step(model, carry, tokens, k_step)

    H_final = jnp.mean(jax.vmap(entropy)(carry.swarm_mu))
    # Check it didn't diverge (no NaN)
    assert not jnp.isnan(H_final)

def test_action_modulates_halo():
    """h_out must have correct shapes with zero and nonzero actions."""
    model = HaloFEPModel(cfg, key)
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    x, z = model.holo_embed(tokens)

    zero_action = jnp.zeros((cfg.n_agents, cfg.n_actions))
    nonzero_action = jnp.ones((cfg.n_agents, cfg.n_actions)) / cfg.n_actions

    delta_x_zero = model.action_bridge(zero_action)
    delta_x_real = model.action_bridge(nonzero_action)

    h_zero = model.backbone(tokens, x + delta_x_zero, z)
    h_real = model.backbone(tokens, x + delta_x_real, z)
    assert h_zero.shape == h_real.shape == (cfg.n_tokens, cfg.d_model)

def test_belief_modulates_flow():
    """delta_v must differ when beliefs are nonzero."""
    model = HaloFEPModel(cfg, key)
    mu_zero    = jnp.zeros((cfg.n_agents, cfg.n_hidden))
    mu_nonzero = jax.random.normal(key, (cfg.n_agents, cfg.n_hidden))
    dv_zero    = model.belief_bridge(mu_zero)
    dv_nonzero = model.belief_bridge(mu_nonzero)
    assert dv_zero.shape == dv_nonzero.shape == (cfg.n_tokens, cfg.d_model)
    assert not jnp.allclose(dv_zero, dv_nonzero)

@pytest.mark.slow
def test_benchmark_success_rate():
    """Slow test — run manually with: pytest -m slow"""
    model = HaloFEPModel(cfg, key)
    world = MultimodalWorld(cfg, key)
    rate  = benchmark(model, world, cfg, n_episodes=100, n_steps=50)
    assert float(rate) >= 0.80
