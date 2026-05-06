# halo_fep/training/bootstrap.py
"""Phase 0 bootstrap: pre-train HALO+FEP on MultimodalWorld, save checkpoint.

Optionally uses Wikipedia topic data (requires `pip install datasets`) and
WN18RR hyperbolic pre-training. Multi-scale SSM training is always enabled.

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


def _multiscale_elbo_loss(
    model: HaloFEPModel,
    carry,
    tokens: jnp.ndarray,
    key: jnp.ndarray,
    strides: tuple[int, ...] = (1, 4),
) -> tuple[jnp.ndarray, dict]:
    """Compute ELBO at multiple temporal strides and return the mean.

    Stride 1 = original sequence (fine-grained patterns).
    Stride 4 = every 4th token subsampled then zero-padded (coarse patterns).

    Training at both scales simultaneously gives the SSM blocks richer
    temporal representations — analogous to multi-resolution wavelet analysis.

    Args:
        model: HaloFEPModel.
        carry: HaloFEPCarry (belief state).
        tokens: (n_tokens, d_model) float32 input.
        key: JAX PRNG key.
        strides: Tuple of stride values. Each > 0.

    Returns:
        (mean_loss, aux_from_first_stride)
    """
    n_tokens = tokens.shape[0]
    total_loss = jnp.zeros(())
    first_aux = None

    for stride in strides:
        # Subsample: take every `stride`-th token
        indices = jnp.arange(0, n_tokens, stride)
        sub = tokens[indices]                                   # (n_tokens//stride, d_model)
        # Pad back to n_tokens with zeros
        pad_len = n_tokens - sub.shape[0]
        padded  = jnp.concatenate(
            [sub, jnp.zeros((pad_len, tokens.shape[1]), dtype=jnp.float32)],
            axis=0,
        )                                                       # (n_tokens, d_model)
        key, sk = jax.random.split(key)
        loss_i, aux_i = unified_elbo_loss(model, carry, padded, sk)
        total_loss = total_loss + loss_i
        if first_aux is None:
            first_aux = aux_i

    return total_loss / len(strides), first_aux


def run_bootstrap(
    cfg: HaloFEPConfig,
    n_pretrain_steps: int = 5_000,
    n_synthetic_episodes: int = 100,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
    seed: int = 42,
    use_wikipedia: bool = False,
    use_wn18rr: bool = False,
    multiscale_strides: tuple[int, ...] = (1, 4),
) -> HaloFEPModel:
    """Run Phase 0 bootstrap pre-training.

    Args:
        cfg: Model config (n_tokens must match heartbeat config, typically 32).
        n_pretrain_steps: SGD steps on synthetic/Wikipedia data.
        n_synthetic_episodes: Warm-up rollout steps after pre-training.
        checkpoint_dir: Directory to save model checkpoint.
        seed: Random seed for reproducibility.
        use_wikipedia: If True, replace synthetic data with WikiText-103 topic
                       samples (requires `pip install datasets`).
        use_wn18rr: If True, pre-train HoloEmbedding on WN18RR Poincare loss
                    before the main training loop (requires `pip install datasets`).
        multiscale_strides: Stride tuple for multi-scale ELBO loss.
    """
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    model = HaloFEPModel(cfg, k1)
    carry = model.init_carry(key)
    opt   = optax.adam(cfg.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # --- Optional: WN18RR hyperbolic pre-training ---
    if use_wn18rr:
        from halo_fep.training.hyperbolic_pretrain import run_hyperbolic_pretrain
        log.info("Running WN18RR hyperbolic pre-training on HoloEmbedding...")
        model = run_hyperbolic_pretrain(model, cfg, key=k2)

    # --- Main pre-training loop ---
    if use_wikipedia:
        from halo_fep.training.topic_bootstrap import iter_wikipedia_token_batches
        log.info(f"Bootstrap: {n_pretrain_steps} steps on WikiText-103 topic data.")
        token_iter = iter_wikipedia_token_batches(cfg, seed=seed)
    else:
        world = MultimodalWorld(cfg, k2)
        log.info(f"Bootstrap: {n_pretrain_steps} steps on MultimodalWorld.")

    for step in range(n_pretrain_steps):
        key, sk1, sk2 = jax.random.split(key, 3)

        if use_wikipedia:
            tokens = jnp.array(next(token_iter))
        else:
            eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
            tokens2, _ = world.sample(eta, sk2)
            tokens = _pad_tokens(tokens2, cfg.n_tokens)

        # Multi-scale ELBO loss
        (loss, _), grads = eqx.filter_value_and_grad(
            _multiscale_elbo_loss, has_aux=True
        )(model, carry, tokens, sk2, multiscale_strides)

        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)

        if step % 500 == 0:
            log.info(f"Step {step}/{n_pretrain_steps} | loss={float(loss):.4f}")

    # --- Warm-up rollouts ---
    log.info(f"Running {n_synthetic_episodes} warm-up rollout episodes.")
    world_warmup = MultimodalWorld(cfg, k2)
    for ep in range(n_synthetic_episodes):
        key, sk1, sk2 = jax.random.split(key, 3)
        eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
        tokens2, _ = world_warmup.sample(eta, sk2)
        tokens = _pad_tokens(tokens2, cfg.n_tokens)
        carry, _ = halo_fep_step(model, carry, tokens, sk2)

    save_checkpoint(model, checkpoint_dir)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)
    run_bootstrap(cfg)
