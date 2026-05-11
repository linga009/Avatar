#!/usr/bin/env python3
"""Train HoloBiont 3.0 as a language model on TinyStories.

Tests VRAM fit, then trains if it fits.

Usage:
    python train_tinystories.py --data D:/MLM/Manifold_Language_Model/processed.parquet
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

log = logging.getLogger("train_lm")


def build_tokenizer(stories: list[str], vocab_size: int = 8000):
    """Build a simple word-level tokenizer from stories."""
    from collections import Counter

    # Count all words
    word_counts = Counter()
    for story in stories[:50000]:  # sample for speed
        words = story.lower().split()
        word_counts.update(words)

    # Top vocab_size-3 words + special tokens
    special = ["<pad>", "<unk>", "<eos>"]
    top_words = [w for w, _ in word_counts.most_common(vocab_size - len(special))]
    vocab = special + top_words

    word2id = {w: i for i, w in enumerate(vocab)}
    return word2id, vocab


def tokenize(text: str, word2id: dict, max_len: int = 128) -> list[int]:
    """Tokenize text to IDs."""
    unk_id = word2id.get("<unk>", 1)
    eos_id = word2id.get("<eos>", 2)
    words = text.lower().split()[:max_len - 1]
    ids = [word2id.get(w, unk_id) for w in words] + [eos_id]
    # Pad
    while len(ids) < max_len:
        ids.append(0)  # pad
    return ids[:max_len]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="D:/MLM/Manifold_Language_Model/processed.parquet")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--test-only", action="store_true", help="Just test VRAM fit, don't train")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    backend = jax.default_backend()
    log.info(f"JAX backend: {backend}, devices: {jax.devices()}")

    # --- Load data ---
    log.info(f"Loading TinyStories from {args.data}...")
    import pyarrow.parquet as pq
    table = pq.read_table(args.data, columns=["story"])
    stories = [str(table.column("story")[i]) for i in range(min(len(table), 100000))]
    log.info(f"Loaded {len(stories)} stories")

    # --- Build tokenizer ---
    from halo3.config import Halo3Config
    cfg = Halo3Config()
    log.info(f"Config: d_model={cfg.d_model}, vocab={cfg.vocab_size}, max_seq={cfg.max_seq_len}")

    word2id, vocab = build_tokenizer(stories, cfg.vocab_size)
    log.info(f"Tokenizer: {len(vocab)} words")

    # --- Build model ---
    from halo3.backbone import Halo3Backbone
    from halo3.lorentz_embedding import LorentzEmbedding
    from halo3.lm_head import LanguageModelHead, lm_loss

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    lm_head = LanguageModelHead(cfg, k1)
    backbone = Halo3Backbone(cfg, k2)
    lorentz_embed = LorentzEmbedding(cfg, k3)

    n_params_lm = sum(p.size for p in jax.tree_util.tree_leaves(lm_head) if hasattr(p, 'size'))
    n_params_bb = sum(p.size for p in jax.tree_util.tree_leaves(backbone) if hasattr(p, 'size'))
    n_params_le = sum(p.size for p in jax.tree_util.tree_leaves(lorentz_embed) if hasattr(p, 'size'))
    total = n_params_lm + n_params_bb + n_params_le
    log.info(f"Params: LM head={n_params_lm/1e6:.1f}M, backbone={n_params_bb/1e6:.1f}M, "
             f"lorentz={n_params_le/1e6:.2f}M, total={total/1e6:.1f}M")

    # --- Test VRAM with one forward+backward step ---
    log.info("Testing VRAM fit with one forward+backward step...")
    story = stories[0]
    token_ids = jnp.array(tokenize(story, word2id, cfg.max_seq_len), dtype=jnp.int32)

    # Forward
    loss, n_correct = lm_loss(lm_head, backbone, token_ids, lorentz_embed, key)
    log.info(f"Forward pass: loss={float(loss):.4f}, correct={int(n_correct)}/{cfg.max_seq_len-1}")

    # Backward
    def total_loss(lm_h, bb, le):
        l, _ = lm_loss(lm_h, bb, token_ids, le, key)
        return l

    grads = jax.grad(total_loss, argnums=(0, 1, 2))(lm_head, backbone, lorentz_embed)
    log.info("Backward pass: gradients computed successfully")

    # Check VRAM
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    log.info(f"VRAM after forward+backward: {result.stdout.strip()}")

    if args.test_only:
        log.info("VRAM test passed. Use --steps N to train.")
        return

    # --- Train ---
    log.info(f"Training for {args.steps} steps...")
    opt = optax.adam(3e-4)

    # Combine all params for optimizer
    all_params = (lm_head, backbone, lorentz_embed)
    opt_state = opt.init(jax.tree_util.tree_map(lambda x: x, all_params))

    @jax.jit
    def train_step(lm_h, bb, le, token_ids, opt_state, key):
        def loss_fn(lm_h, bb, le):
            l, nc = lm_loss(lm_h, bb, token_ids, le, key)
            return l, nc
        (loss, n_correct), grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True)(lm_h, bb, le)
        updates, new_opt_state = opt.update(grads, opt_state, (lm_h, bb, le))
        lm_h, bb, le = jax.tree_util.tree_map(lambda p, u: p + u, (lm_h, bb, le), updates)
        return lm_h, bb, le, new_opt_state, loss, n_correct

    log.info("JIT compiling training step (first call will be slow)...")
    t0 = time.time()

    for step in range(args.steps):
        idx = step % len(stories)
        token_ids = jnp.array(tokenize(stories[idx], word2id, cfg.max_seq_len), dtype=jnp.int32)
        key, sk = jax.random.split(key)

        lm_head, backbone, lorentz_embed, opt_state, loss, n_correct = train_step(
            lm_head, backbone, lorentz_embed, token_ids, opt_state, sk
        )

        if step % 100 == 0:
            acc = int(n_correct) / (cfg.max_seq_len - 1) * 100
            elapsed = time.time() - t0
            log.info(f"Step {step}/{args.steps} | loss={float(loss):.4f} | "
                     f"acc={acc:.1f}% | {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    log.info(f"Training complete: {args.steps} steps in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # --- Test generation ---
    log.info("Testing generation...")
    prompt = "once there was a"
    prompt_ids = jnp.array(tokenize(prompt, word2id, 10), dtype=jnp.int32)[:5]

    h = lm_head.embed(prompt_ids)
    x, z = lorentz_embed(h)
    h_out = backbone(h, x, z)
    logits = lm_head.project(h_out)
    next_token = int(jnp.argmax(logits[-1]))
    next_word = vocab[next_token] if next_token < len(vocab) else "<unk>"
    log.info(f"Prompt: '{prompt}' -> next word: '{next_word}'")


if __name__ == "__main__":
    main()
