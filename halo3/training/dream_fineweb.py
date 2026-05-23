"""FineWeb dream phase — Phase 4 of body dreaming.

Runs n_steps CLion gradient steps on sequentially-cursored FineWeb-Edu rows.
Same JIT pattern as dream_replay.py: single @eqx.filter_jit defined
ONCE outside the loop; scale passed as jnp.float32() argument.

Uses a persistent cursor (data/checkpoints/fineweb_cursor.json) so each
dream cycle sees fresh texts — no repetitions until the full corpus wraps.
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
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_CURSOR_PATH = "data/checkpoints/fineweb_cursor.json"
_MIN_SCORE = 3


def _load_cursor() -> dict:
    if os.path.exists(_CURSOR_PATH):
        with open(_CURSOR_PATH) as f:
            return json.load(f)
    return {"file_idx": 0, "row_group_idx": 0, "row_offset": 0}


def _save_cursor(cursor: dict) -> None:
    os.makedirs(os.path.dirname(_CURSOR_PATH), exist_ok=True)
    with open(_CURSOR_PATH, "w") as f:
        json.dump(cursor, f)


def _sample_texts_cursor(files: list[str], n: int) -> list[str]:
    """Read n texts from parquet files starting at the persisted cursor position.

    Streams row groups one at a time — never loads the full corpus.
    Wraps around when the dataset is exhausted.
    """
    cursor = _load_cursor()
    file_idx = cursor["file_idx"] % len(files)
    rg_idx = cursor["row_group_idx"]
    row_off = cursor["row_offset"]

    texts: list[str] = []
    files_visited = 0

    while len(texts) < n and files_visited <= len(files):
        path = files[file_idx % len(files)]
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            log.warning(f"  FineWeb cursor: skipping {os.path.basename(path)}: {e}")
            file_idx += 1
            rg_idx = 0
            row_off = 0
            files_visited += 1
            continue

        while rg_idx < pf.num_row_groups and len(texts) < n:
            batch = pf.read_row_group(rg_idx, columns=["text", "int_score"])
            d = batch.to_pydict()
            found_enough = False
            for i in range(row_off, len(d["text"])):
                if len(texts) >= n:
                    row_off = i
                    found_enough = True
                    break
                text = d["text"][i]
                score = d["int_score"][i] or 0
                if score >= _MIN_SCORE and text and text.strip():
                    texts.append(text)
            if not found_enough:
                # exhausted this row group — advance to next
                rg_idx += 1
                row_off = 0
            del batch, d

        # Only advance to next file if we truly exhausted all row groups
        if rg_idx >= pf.num_row_groups:
            file_idx += 1
            rg_idx = 0
            row_off = 0
            files_visited += 1

    _save_cursor({"file_idx": file_idx % len(files),
                  "row_group_idx": rg_idx,
                  "row_offset": row_off})
    log.info(f"  FineWeb cursor: read {len(texts)} texts, next cursor: file={file_idx % len(files)} rg={rg_idx} row={row_off}")
    return texts


def fineweb_dream_phase(
    model,
    parquet_dir: str = "data/fineweb",
    n_steps: int = 40,
    lr: float = 5e-6,
    seed: int = 99,
) -> tuple[any, dict]:
    """Train body on n_steps FineWeb-Edu rows using CLion optimizer.

    Returns (updated_model, info_dict).
    Skips silently (returns model unchanged) if no Parquet files found.
    Uses persistent cursor — no repetitions until full corpus wraps.
    """
    files = sorted(glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True))
    if not files:
        log.info("FineWeb dream: no Parquet files found — skipping phase 4")
        return model, {"fineweb_steps": 0, "fineweb_loss": 0.0}

    from halo3.perception.native_embedder import NativeEmbedder
    from halo3.loss import halo3_loss
    from halo3.training.dream_replay import scale_by_clion

    log.info(f"  FineWeb dream: cursor-read {n_steps} texts (lr={lr})...")
    texts = _sample_texts_cursor(files, n_steps)
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
