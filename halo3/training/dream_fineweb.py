"""FineWeb dream phase — Phase 4 of body dreaming.

v3.11: Free Energy-guided active learning replaces sequential cursor.
ActiveSampler uses BS valuation to pick topics, then forward-only FE
to filter candidates into the zone of proximal development.
"""
from __future__ import annotations
import gc
import glob
import json
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
    n_steps: int = 10,
    lr: float = 5e-6,
    seed: int = 99,
    bs_state_path: str = "data/dream_training/bs_state.json",
    index_path: str = "data/fineweb/topic_index.json",
) -> tuple[any, dict]:
    """Train body on FE-selected FineWeb-Edu texts using CLion optimizer."""
    from halo3.perception.native_embedder import NativeEmbedder
    from halo3.loss import halo3_loss
    from halo3.training.dream_replay import scale_by_clion

    # Load topic index (auto-build if missing)
    if not os.path.exists(index_path):
        parquet_files = glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True)
        if not parquet_files:
            log.info("FineWeb dream: no parquet files found — skipping phase 4")
            return model, {"fineweb_steps": 0, "fineweb_loss": 0.0}
        log.warning("FineWeb dream: no topic index — building now...")
        from halo3.perception.topic_index import TopicIndex
        TopicIndex.build(parquet_dir, index_path)

    from halo3.perception.topic_index import TopicIndex
    topic_index = TopicIndex(index_path, parquet_dir)

    # Load BS state for topic ranking
    if os.path.exists(bs_state_path):
        from halo3.psyche.volatility import VolatilitySurface
        vol_surface = VolatilitySurface.load_state(bs_state_path)
        log.info("FineWeb dream: loaded BS state for active sampling")
    else:
        from halo3.psyche.volatility import VolatilitySurface
        vol_surface = VolatilitySurface()
        log.info("FineWeb dream: no BS state — using default priors")

    # Select curriculum via ActiveSampler
    embedder = NativeEmbedder(model.cfg.d_model, n_tokens=model.cfg.n_tokens)
    key = jax.random.PRNGKey(seed)

    from halo3.training.active_sampler import select_curriculum
    key, sample_key = jax.random.split(key)
    carry = model.init_carry(sample_key)

    texts = select_curriculum(
        model=model,
        carry=carry,
        topic_index=topic_index,
        volatility_surface=vol_surface,
        embedder=embedder,
        n_candidates=min(50, n_steps * 5),
        n_train=n_steps,
        key=sample_key,
    )

    if not texts:
        log.warning("FineWeb dream: ActiveSampler returned no texts — skipping")
        return model, {"fineweb_steps": 0, "fineweb_loss": 0.0}

    log.info(f"FineWeb dream: ActiveSampler selected {len(texts)} texts")

    # Embed selected texts
    log.info(f"  Pre-embedding {len(texts)} texts on CPU...")
    token_tensors = [embedder.texts_to_tokens([t], model.cfg.n_tokens) for t in texts]
    del embedder
    gc.collect()
    jax.clear_caches()

    # CLion optimizer
    opt = optax.chain(
        optax.clip_by_global_norm(0.1),
        scale_by_clion(b1=0.9),
        optax.scale(-lr),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

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
            if loss_val != loss_val:
                return model, opt_state, loss
            leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
            if any(bool(jnp.any(jnp.isnan(leaf))) for leaf in leaves):
                log.warning("  FineWeb: NaN weights after step — skipping")
                return model, opt_state, loss
            return new_model, new_opt_state, loss
        except Exception as e:
            log.warning(f"  FineWeb step exception: {e}")
            return model, opt_state, 0.0

    log.info(f"  Phase 4: FineWeb active learning ({len(token_tensors)} steps, scale=0.05)...")
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
    log.info(f"  FineWeb phase done: {completed}/{len(token_tensors)} steps | avg_loss={avg_loss:.2e}")
    return model, {"fineweb_steps": completed, "fineweb_loss": avg_loss}
