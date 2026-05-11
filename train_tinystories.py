#!/usr/bin/env python3
"""Train HoloBiont 3.0 on TinyStories — full pipeline.

All 4 improvements:
1. 50K steps (configurable)
2. Gradient accumulation (effective batch=4, zero extra VRAM)
3. Full 781K stories (streaming from parquet)
4. BPE tokenizer via sentencepiece

Usage:
    docker run --gpus all -v D:/MLM/Manifold_Language_Model:/data \
      halo3-train python3 train_tinystories.py --data /data/processed.parquet --steps 50000
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
import tempfile

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

log = logging.getLogger("train_lm")


def build_bpe_tokenizer(stories: list[str], vocab_size: int = 8000, model_prefix: str = "data/tokenizer"):
    """Train a BPE tokenizer on stories using sentencepiece."""
    import sentencepiece as spm

    model_path = f"{model_prefix}.model"
    if os.path.exists(model_path):
        log.info(f"Loading existing BPE tokenizer from {model_path}")
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp

    log.info(f"Training BPE tokenizer on {len(stories)} stories (vocab={vocab_size})...")
    os.makedirs(os.path.dirname(model_prefix) if os.path.dirname(model_prefix) else ".", exist_ok=True)

    # Write stories to temp file for sentencepiece
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    for s in stories[:100000]:
        tmp.write(s.strip() + "\n")
    tmp.close()

    spm.SentencePieceTrainer.Train(
        input=tmp.name,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        character_coverage=0.9995,
        max_sentence_length=4096,
    )
    os.unlink(tmp.name)

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    log.info(f"BPE tokenizer trained: {sp.GetPieceSize()} pieces")
    return sp


def tokenize_bpe(text: str, sp, max_len: int = 128) -> list[int]:
    """Tokenize with BPE, pad/truncate to max_len."""
    ids = sp.Encode(text)[:max_len - 1] + [3]  # add EOS
    while len(ids) < max_len:
        ids.append(0)  # pad
    return ids[:max_len]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="D:/MLM/Manifold_Language_Model/processed.parquet")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/halo3_lm")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    backend = jax.default_backend()
    log.info(f"JAX backend: {backend}, devices: {jax.devices()}")

    # --- Load stories ---
    log.info(f"Loading stories from {args.data}...")
    import pyarrow.parquet as pq
    table = pq.read_table(args.data, columns=["story"])
    n_stories = len(table)
    log.info(f"Dataset: {n_stories:,} stories")

    # Stream stories one at a time (no need to hold all in RAM)
    def get_story(idx):
        return str(table.column("story")[idx % n_stories])

    # --- BPE tokenizer ---
    from halo3.config import Halo3Config
    cfg = Halo3Config()

    # Sample stories for tokenizer training
    sample_stories = [get_story(i) for i in range(min(n_stories, 100000))]
    sp = build_bpe_tokenizer(sample_stories, cfg.vocab_size)

    log.info(f"Config: d_model={cfg.d_model}, vocab={cfg.vocab_size}, seq={cfg.max_seq_len}")

    # --- Build model ---
    from halo3.backbone import Halo3Backbone
    from halo3.lorentz_embedding import LorentzEmbedding
    from halo3.lm_head import LanguageModelHead, lm_loss

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    lm_head = LanguageModelHead(cfg, k1)
    backbone = Halo3Backbone(cfg, k2)
    lorentz_embed = LorentzEmbedding(cfg, k3)

    total_params = sum(p.size for p in jax.tree_util.tree_leaves((lm_head, backbone, lorentz_embed)) if hasattr(p, 'size'))
    log.info(f"Total params: {total_params/1e6:.1f}M")

    # --- Optimizer with warmup ---
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=3e-4,
        warmup_steps=1000, decay_steps=args.steps, end_value=3e-5,
    )
    opt = optax.adam(schedule)
    opt_state = opt.init(jax.tree_util.tree_map(lambda x: x, (lm_head, backbone, lorentz_embed)))

    # --- JIT training step ---
    @jax.jit
    def train_step(lm_h, bb, le, token_ids, key):
        def loss_fn(lm_h, bb, le):
            l, nc = lm_loss(lm_h, bb, token_ids, le, key)
            return l, nc
        (loss, n_correct), grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(lm_h, bb, le)
        return loss, n_correct, grads

    @jax.jit
    def apply_grads(lm_h, bb, le, grads, opt_state):
        updates, new_opt_state = opt.update(grads, opt_state, (lm_h, bb, le))
        lm_h, bb, le = jax.tree_util.tree_map(lambda p, u: p + u, (lm_h, bb, le), updates)
        return lm_h, bb, le, new_opt_state

    # --- Training loop with gradient accumulation ---
    log.info(f"Training: {args.steps} steps, accum={args.accum}, effective_batch={args.accum}")
    log.info(f"Stories: {n_stories:,} | BPE vocab: {sp.GetPieceSize()}")
    log.info("JIT compiling (first step will be slow)...")

    t0 = time.time()
    accum_grads = None
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for step in range(args.steps):
        # Get story and tokenize
        story = get_story(step)
        token_ids = jnp.array(tokenize_bpe(story, sp, cfg.max_seq_len), dtype=jnp.int32)
        key, sk = jax.random.split(key)

        # Forward + backward (no optimizer step yet)
        loss, n_correct, grads = train_step(lm_head, backbone, lorentz_embed, token_ids, sk)

        # Accumulate gradients
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = jax.tree_util.tree_map(lambda a, g: a + g, accum_grads, grads)

        total_loss += float(loss)
        total_correct += int(n_correct)
        total_tokens += cfg.max_seq_len - 1

        # Apply accumulated gradients every accum steps
        if (step + 1) % args.accum == 0:
            # Average gradients
            accum_grads = jax.tree_util.tree_map(lambda g: g / args.accum, accum_grads)
            lm_head, backbone, lorentz_embed, opt_state = apply_grads(
                lm_head, backbone, lorentz_embed, accum_grads, opt_state
            )
            accum_grads = None

        # Log
        if (step + 1) % args.log_every == 0:
            avg_loss = total_loss / args.log_every
            avg_acc = total_correct / total_tokens * 100
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            eta = (args.steps - step - 1) / steps_per_sec / 60

            log.info(
                f"Step {step+1}/{args.steps} | loss={avg_loss:.0f} | "
                f"acc={avg_acc:.1f}% | {steps_per_sec:.1f} steps/s | ETA {eta:.0f}min"
            )
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else ".", exist_ok=True)
            eqx.tree_serialise_leaves(f"{args.checkpoint}_lm.eqx", lm_head)
            eqx.tree_serialise_leaves(f"{args.checkpoint}_bb.eqx", backbone)
            eqx.tree_serialise_leaves(f"{args.checkpoint}_le.eqx", lorentz_embed)
            log.info(f"Checkpoint saved to {args.checkpoint}_*.eqx")

    elapsed = time.time() - t0
    log.info(f"Training complete: {args.steps} steps in {elapsed/60:.1f} min")

    # --- Test generation ---
    log.info("Testing generation...")
    prompts = [
        "once there was a",
        "the little girl",
        "he was very",
        "she loved to",
        "one day the",
    ]
    for prompt in prompts:
        ids = sp.Encode(prompt)
        if len(ids) < 2:
            continue
        token_ids = jnp.array(ids, dtype=jnp.int32)
        h = lm_head.embed(token_ids)
        x, z = lorentz_embed(h)
        h_out = backbone(h, x, z)
        logits = lm_head.project(h_out)

        # Top-5 predictions for last token
        top5 = jnp.argsort(logits[-1])[-5:][::-1]
        words = [sp.IdToPiece(int(i)) for i in top5]
        log.info(f"  '{prompt}' -> {words}")

    # Final save
    eqx.tree_serialise_leaves(f"{args.checkpoint}_lm.eqx", lm_head)
    eqx.tree_serialise_leaves(f"{args.checkpoint}_bb.eqx", backbone)
    eqx.tree_serialise_leaves(f"{args.checkpoint}_le.eqx", lorentz_embed)
    log.info(f"Final checkpoint saved.")


if __name__ == "__main__":
    main()
