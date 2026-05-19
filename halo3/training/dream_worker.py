"""Subprocess worker for body dreaming — runs in total memory isolation.

Spawned by main.py as a separate Python process.  When this process exits,
the OS reclaims ALL GPU memory, XLA caches, and CPU allocations — guaranteed.
This is far more reliable than gc.collect() + jax.clear_caches().

Protocol:
  1. Parent saves model checkpoint + flushes episode store
  2. Parent frees GPU memory (del model)
  3. Parent spawns this worker via subprocess.run()
  4. Worker loads model, runs dream, saves updated model
  5. Worker writes dream_info.json for parent to read
  6. Worker exits → OS reclaims everything
  7. Parent reloads model from checkpoint
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("dream_worker")

    parser = argparse.ArgumentParser(description="Body dream subprocess")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path (without .eqx)")
    parser.add_argument("--output", required=True, help="Output checkpoint path (without .eqx)")
    parser.add_argument("--replay-steps", type=int, default=10)
    parser.add_argument("--recombine-steps", type=int, default=5)
    parser.add_argument("--imagine-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--fineweb-steps", type=int, default=10,
                        help="FineWeb batch steps per dream (0 = disabled)")
    args = parser.parse_args()

    # --- XLA compilation cache (same as parent) ---
    import jax
    xla_cache = os.path.abspath(os.path.join("data", "xla_cache"))
    os.makedirs(xla_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", xla_cache)
    log.info(f"XLA cache: {xla_cache}")

    backend = jax.default_backend()
    log.info(f"Dream worker started — JAX backend: {backend}, devices: {jax.devices()}")

    # --- Load model ---
    from halo3.config import Halo3Config
    from halo3.training.bootstrap import load_checkpoint, save_checkpoint

    if backend in ("gpu", "cuda"):
        cfg = Halo3Config()
    else:
        cfg = Halo3Config(
            d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
            d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
            n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
            mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
            meta_n_hidden=4, meta_n_actions=2, meta_k=3,
            max_cache=8, island_size=4,
        )

    log.info(f"Loading model from {args.checkpoint}.eqx")
    model = load_checkpoint(cfg, args.checkpoint)

    # --- Load episodes ---
    from halo3.memory.episode_store import EpisodeStore
    memory = EpisodeStore()
    episodes = memory.get_high_confidence(threshold=0.0)
    log.info(f"Loaded {len(episodes)} episodes for dreaming")

    # --- Dream ---
    from halo3.training.dream_replay import dream_replay_physics
    model, dream_info = dream_replay_physics(
        model, episodes,
        n_replay_steps=args.replay_steps,
        n_recombine_steps=args.recombine_steps,
        n_imagine_steps=args.imagine_steps,
        lr=args.lr,
    )

    # --- Save ---
    log.info(f"Saving dreamed model to {args.output}.eqx")
    save_checkpoint(model, args.output)

    # --- Phase 4: FineWeb batch training ---
    if args.fineweb_steps > 0:
        from halo3.training.dream_fineweb import fineweb_dream_phase
        log.info(f"  ☽ Phase 4: FineWeb batch ({args.fineweb_steps} steps)...")
        model, fw_info = fineweb_dream_phase(
            model,
            parquet_dir="data/fineweb",
            n_steps=args.fineweb_steps,
            lr=args.lr,
        )
        dream_info.update(fw_info)
        save_checkpoint(model, args.output)
        log.info(f"  ☽ FineWeb phase done: {fw_info}")

    # Write dream info for parent to read
    info_path = args.output + "_dream_info.json"
    with open(info_path, "w") as f:
        json.dump(dream_info, f)
    log.info(f"Dream worker done: {dream_info}")


if __name__ == "__main__":
    main()
