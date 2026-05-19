"""FineWeb dream phase — Phase 4 of body dreaming.

Runs n_steps CLion gradient steps on randomly sampled FineWeb-Edu rows.
Same JIT pattern as dream_replay.py: single @eqx.filter_jit defined
ONCE outside the loop; scale passed as jnp.float32() argument.
"""
from __future__ import annotations
import gc
import glob
import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

log = logging.getLogger(__name__)


def fineweb_dream_phase(
    model,
    parquet_dir: str = "data/fineweb",
    n_steps: int = 40,
    lr: float = 5e-6,
    seed: int = 99,
) -> tuple[any, dict]:
    """Train body on n_steps random FineWeb-Edu rows using CLion optimizer.

    Returns (updated_model, info_dict).
    Skips silently (returns model unchanged) if no Parquet files found.
    """
    files = glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True)
    if not files:
        log.info("FineWeb dream: no Parquet files found — skipping phase 4")
        return model, {"fineweb_steps": 0, "fineweb_loss": 0.0}

    from halo3.perception.parquet_source import ParquetSource
    from halo3.perception.native_embedder import NativeEmbedder
    from halo3.loss import halo3_loss
    from halo3.training.dream_replay import scale_by_clion

    log.info(f"  FineWeb dream: loading source ({n_steps} steps, lr={lr})...")
    source = ParquetSource(parquet_dir)
    texts = source.sample_texts(n_steps)
    del source          # free ~450MB RAM before embedder
    gc.collect()

    embedder = NativeEmbedder(model.cfg.d_model, n_tokens=model.cfg.n_tokens)
    log.info(f"  Pre-embedding {len(texts)} texts on CPU...")
    token_tensors = [embedder.texts_to_tokens([t], model.cfg.n_tokens) for t in texts]
    del embedder        # free ~300MB RAM before optimizer state
    del texts
    gc.collect()
    jax.clear_caches()  # flush XLA buffers before new allocations

    key = jax.random.PRNGKey(seed)
    carry = model.init_carry(key)

    opt = optax.chain(
        optax.clip_by_global_norm(0.1),
        scale_by_clion(b1=0.9),
        optax.scale(-lr),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Single JIT defined ONCE outside loop — prevents XLA recompilation OOM
    @eqx.filter_jit
    def _fineweb_step(model, opt_state_in, carry, tokens, key, scale):
        loss_fn = lambda m: halo3_loss(m, carry, tokens, key)[0] * scale
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state_in,
            eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss

    def _safe_step(model, opt_state, carry, tokens, key, scale):
        try:
            new_model, new_opt_state, loss = _fineweb_step(
                model, opt_state, carry, tokens, key, jnp.float32(scale)
            )
            loss_val = float(loss)
            if loss_val != loss_val:  # NaN check
                return model, opt_state, loss
            leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
            if any(bool(jnp.any(jnp.isnan(leaf))) for leaf in leaves):
                log.warning("  FineWeb: NaN weights after step — skipping")
                return model, opt_state, loss
            return new_model, new_opt_state, loss
        except Exception as e:
            log.warning(f"  FineWeb step exception: {e}")
            return model, opt_state, 0.0

    log.info(f"  Phase 4: FineWeb batch ({n_steps} steps, scale=0.05)...")
    total_loss = 0.0
    completed = 0
    for tokens in token_tensors:
        key, sk = jax.random.split(key)
        model, opt_state, loss = _safe_step(model, opt_state, carry, tokens, sk, 0.05)
        total_loss += float(loss)
        completed += 1

    avg_loss = total_loss / max(completed, 1)
    gc.collect()
    jax.clear_caches()
    log.info(
        f"  FineWeb phase done: {completed}/{n_steps} steps | avg_loss={avg_loss:.2e}"
    )
    return model, {"fineweb_steps": completed, "fineweb_loss": avg_loss}
