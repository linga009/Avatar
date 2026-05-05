# halo_fep/benchmark/eval.py
"""Benchmark evaluation for the Multimodal Goal Inference task.

An episode succeeds when >= 80% of agents infer the correct eta at step n_steps.
Success is defined as argmax(mu_i) == eta for each agent.
"""
import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.benchmark.multimodal_world import MultimodalWorld


def run_episode(
    model: HaloFEPModel,
    world: MultimodalWorld,
    cfg: HaloFEPConfig,
    key: jnp.ndarray,
    n_steps: int = 50,
) -> bool:
    """Run one episode. Returns True if >=80% of agents infer eta correctly."""
    k_eta, k_run = jax.random.split(key)
    eta   = int(jax.random.randint(k_eta, (), 0, cfg.n_hidden))
    carry = model.init_carry(k_run)

    k = k_run
    for _ in range(n_steps):
        k, k_step = jax.random.split(k)
        tokens, _ = world.sample(eta=eta, key=k_step)
        carry, _  = halo_fep_step(model, carry, tokens, k_step)

    predicted = jnp.argmax(carry.swarm_mu, axis=-1)   # (N_agents,)
    success_rate = jnp.mean(predicted == eta)
    return bool(success_rate >= 0.80)


def benchmark(
    model: HaloFEPModel,
    world: MultimodalWorld,
    cfg: HaloFEPConfig,
    n_episodes: int = 100,
    n_steps: int = 50,
) -> float:
    """Run n_episodes and return fraction of successful episodes."""
    keys     = jax.random.split(jax.random.PRNGKey(0), n_episodes)
    results  = [run_episode(model, world, cfg, k, n_steps=n_steps) for k in keys]
    return float(sum(results)) / n_episodes
