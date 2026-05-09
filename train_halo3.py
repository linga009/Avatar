#!/usr/bin/env python3
"""GPU training entrypoint for HoloBiont 3.0 Physics Engine."""
from __future__ import annotations
import argparse
import logging
import sys
import time

log = logging.getLogger("train_halo3")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="HoloBiont 3.0 bootstrap")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default="data/checkpoints/halo3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import jax
    backend = jax.default_backend()
    log.info(f"JAX backend: {backend}")
    log.info(f"JAX devices: {jax.devices()}")

    from halo3.config import Halo3Config

    if backend in ("gpu", "cuda"):
        log.info("GPU detected — full scale (1.7B params, ~4.5 GB).")
        cfg = Halo3Config()
    else:
        log.warning("No GPU — small config for CPU.")
        cfg = Halo3Config(
            d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
            d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
            n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
            mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
            meta_n_hidden=4, meta_n_actions=2, meta_k=3,
            max_cache=8, island_size=4,
        )

    log.info(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
             f"d_state={cfg.d_state}, mera_bond={cfg.mera_bond_dim}")

    from halo3.training.bootstrap import run_bootstrap

    t0 = time.time()
    log.info(f"Starting bootstrap: {args.steps} steps")

    model = run_bootstrap(
        cfg, n_steps=args.steps,
        checkpoint_dir=args.checkpoint, seed=args.seed,
    )

    elapsed = time.time() - t0
    log.info(f"Bootstrap complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Validation
    import jax.numpy as jnp
    from halo3.loss import halo3_loss

    key = jax.random.PRNGKey(99)
    carry = model.init_carry(key)
    losses = []
    for i in range(5):
        key, sk = jax.random.split(key)
        tokens = jax.random.normal(sk, (cfg.n_tokens, cfg.d_model))
        loss, _ = halo3_loss(model, carry, tokens, sk)
        losses.append(float(loss))

    log.info(f"Validation losses: {[f'{v:.3f}' for v in losses]}")

    if any(jnp.isnan(v) for v in losses):
        log.error("NaN detected!")
        sys.exit(1)

    log.info("HoloBiont 3.0 training complete.")


if __name__ == "__main__":
    main()
