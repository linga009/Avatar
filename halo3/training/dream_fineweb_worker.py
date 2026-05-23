"""Subprocess worker for FineWeb Phase 4 dreaming — runs in total memory isolation.

Spawned by main.py as a SEPARATE process AFTER the body dream worker exits.
This guarantees that XLA caches from phases 1-3 are fully reclaimed by the OS
before Phase 4 allocates its own JIT compilations.

Protocol:
  1. Body dream worker exits (phases 1-3 done, checkpoint saved)
  2. Parent spawns this worker
  3. Worker loads checkpoint, runs FineWeb batch training, saves updated model
  4. Worker writes fineweb_dream_info.json for parent to read
  5. Worker exits -> OS reclaims everything
"""
from __future__ import annotations
import argparse
import json
import logging
import os


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("fineweb_worker")

    parser = argparse.ArgumentParser(description="FineWeb dream subprocess")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path (without .eqx)")
    parser.add_argument("--output", required=True, help="Output checkpoint path (without .eqx)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    args = parser.parse_args()

    import jax
    xla_cache = os.path.abspath(os.path.join("data", "xla_cache"))
    os.makedirs(xla_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", xla_cache)

    backend = jax.default_backend()
    log.info(f"FineWeb worker started — JAX backend: {backend}, devices: {jax.devices()}")

    from halo3.config import Halo3Config
    from halo3.training.bootstrap import load_checkpoint, save_checkpoint

    cfg = Halo3Config()
    log.info(f"Loading model from {args.checkpoint}.eqx")
    model = load_checkpoint(cfg, args.checkpoint)

    from halo3.training.dream_fineweb import fineweb_dream_phase
    log.info(f"FineWeb batch: {args.steps} steps, lr={args.lr}")
    model, fw_info = fineweb_dream_phase(
        model,
        parquet_dir="data/fineweb",
        n_steps=args.steps,
        lr=args.lr,
    )

    save_checkpoint(model, args.output)
    log.info(f"FineWeb dream done: {fw_info}")

    info_path = os.path.join(os.path.dirname(args.output), "fineweb_dream_info.json")
    with open(info_path, "w") as f:
        json.dump(fw_info, f)


if __name__ == "__main__":
    main()
