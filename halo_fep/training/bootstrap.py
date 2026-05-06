# halo_fep/training/bootstrap.py
"""Phase 0 bootstrap: pre-train HALO+FEP on MultimodalWorld, save checkpoint.

Usage:
    python -m halo_fep.training.bootstrap
"""
from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.loss import unified_elbo_loss
from halo_fep.benchmark.multimodal_world import MultimodalWorld

log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "data/checkpoints/bootstrap"


def save_checkpoint(model: HaloFEPModel, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    eqx.tree_serialise_leaves(path + ".eqx", model)
    log.info(f"Checkpoint saved to {path}.eqx")


def load_checkpoint(cfg: HaloFEPConfig, path: str) -> HaloFEPModel:
    template = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    model    = eqx.tree_deserialise_leaves(path + ".eqx", template)
    log.info(f"Checkpoint loaded from {path}.eqx")
    return model


def _pad_tokens(tokens_2: jnp.ndarray, n_tokens: int) -> jnp.ndarray:
    """Pad (2, d_model) to (n_tokens, d_model) with zeros."""
    d_model = tokens_2.shape[1]
    pad = jnp.zeros((n_tokens - 2, d_model), dtype=jnp.float32)
    return jnp.concatenate([tokens_2, pad], axis=0)


def run_bootstrap(
    cfg: HaloFEPConfig,
    n_pretrain_steps: int = 5_000,
    n_synthetic_episodes: int = 100,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
    seed: int = 42,
) -> HaloFEPModel:
    key   = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    model = HaloFEPModel(cfg, k1)
    world = MultimodalWorld(cfg, k2)
    carry = model.init_carry(key)
    opt   = optax.adam(cfg.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    log.info(f"Bootstrap pre-training: {n_pretrain_steps} steps on MultimodalWorld.")

    for step in range(n_pretrain_steps):
        key, sk1, sk2 = jax.random.split(key, 3)
        eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
        tokens2, _ = world.sample(eta, sk2)
        tokens = _pad_tokens(tokens2, cfg.n_tokens)

        (loss, _), grads = eqx.filter_value_and_grad(unified_elbo_loss, has_aux=True)(
            model, carry, tokens, sk2
        )
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)

        if step % 500 == 0:
            log.info(f"Step {step}/{n_pretrain_steps} | loss={float(loss):.4f}")

    log.info(f"Running {n_synthetic_episodes} synthetic episodes to warm up memory.")
    for ep in range(n_synthetic_episodes):
        key, sk1, sk2 = jax.random.split(key, 3)
        eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
        tokens2, _ = world.sample(eta, sk2)
        tokens = _pad_tokens(tokens2, cfg.n_tokens)
        carry, _ = halo_fep_step(model, carry, tokens, sk2)

    save_checkpoint(model, checkpoint_dir)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)
    run_bootstrap(cfg)
