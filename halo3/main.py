"""HoloBiont Eye — autonomous research monitor.

Runs the Bohmian holomovement engine as a persistent research watcher.
Every tick: fetch web → backbone → Hamiltonian → Kuramoto → findings.

Usage:
    python -m halo3.main
    docker compose up  (with CMD ["python3", "-m", "halo3.main"])
"""
from __future__ import annotations
import datetime
import logging
import os
import signal
import time

import jax
import jax.numpy as jnp
import yaml

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log.info("=" * 60)
    log.info("  HoloBiont Eye — Autonomous Research Monitor")
    log.info("  Bohmian Holomovement Engine v3.0")
    log.info("=" * 60)

    # Load config
    from halo3.config import Halo3Config

    # Use small config for CPU, full for GPU
    backend = jax.default_backend()
    log.info(f"JAX backend: {backend}, devices: {jax.devices()}")

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

    # Load topics
    topics_path = os.path.join(os.path.dirname(__file__), "topics.yaml")
    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topics_cfg = yaml.safe_load(f)
    else:
        topics_cfg = {
            "seed_topics": ["artificial intelligence research"],
            "max_results_per_query": 5,
            "tick_interval": 60,
            "r_exploit_threshold": 0.6,
            "r_explore_threshold": 0.4,
        }

    seed_topics = topics_cfg.get("seed_topics", ["AI research"])
    max_results = topics_cfg.get("max_results_per_query", 5)
    tick_interval = topics_cfg.get("tick_interval", 60)
    r_exploit = topics_cfg.get("r_exploit_threshold", 0.6)
    r_explore = topics_cfg.get("r_explore_threshold", 0.4)

    # Load or init model
    from halo3.model import Halo3Model, halo3_step
    from halo3.training.bootstrap import load_checkpoint

    checkpoint_path = "data/checkpoints/halo3"
    try:
        model = load_checkpoint(cfg, checkpoint_path)
        log.info(f"Loaded checkpoint from {checkpoint_path}.eqx")
    except Exception:
        log.info("No checkpoint found — initializing fresh model")
        model = Halo3Model(cfg, jax.random.PRNGKey(cfg.seed))

    carry = model.init_carry(jax.random.PRNGKey(cfg.seed))
    key = jax.random.PRNGKey(cfg.seed + 1)

    # Init components
    from halo3.perception.pipeline import PerceptionPipeline
    from halo3.perception.interpreter import Interpreter
    from halo3.memory.schema import Episode
    from halo3.memory.episode_store import EpisodeStore
    from halo3.kuramoto import order_parameter

    perception = PerceptionPipeline(cfg.d_model, cfg.n_tokens)
    interpreter = Interpreter(seed_topics, r_exploit, r_explore)
    memory = EpisodeStore()

    log.info(f"Monitoring {len(seed_topics)} topics, tick every {tick_interval}s")
    log.info(f"Topics: {seed_topics}")

    # Signal handling
    shutdown = False

    def _handle_signal(sig, frame):
        nonlocal shutdown
        log.info(f"Signal {sig} — shutting down gracefully")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Heartbeat loop
    tick = 0
    current_query = seed_topics[0]

    log.info("Heartbeat started. Press Ctrl+C to stop.")
    log.info("-" * 60)

    while not shutdown:
        tick += 1
        tick_start = time.time()

        # 1. Perception
        try:
            tokens, texts = perception.perceive(current_query, max_results)
        except Exception as e:
            log.warning(f"Perception failed: {e}")
            tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
            texts = []

        # 2. HoloBiont step
        key, sk = jax.random.split(key)
        try:
            carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)
        except Exception as e:
            log.error(f"halo3_step failed: {e}")
            time.sleep(tick_interval)
            continue

        # 3. Interpret
        result = interpreter.interpret(carry.kuramoto.theta, texts, current_query)
        r_mean = result["r_mean"]
        mode = result["mode"]
        finding = result["finding"]
        current_query = result["next_query"]

        # 4. Store episode
        query_embed = perception.embed_query(current_query)
        episode = Episode(
            query=current_query,
            order_param=r_mean,
            mode=mode,
            finding=finding,
            query_embed=query_embed,
        )
        memory.add(episode)

        # 5. Log
        r_bar = "█" * int(r_mean * 20) + "░" * (20 - int(r_mean * 20))
        log.info(
            f"Tick {tick:4d} | r=[{r_bar}] {r_mean:.3f} | {mode:7s} | "
            f"q=\"{current_query[:50]}\""
        )
        if finding:
            log.info(f"  → FINDING: {finding[:100]}")

        # 6. Timing
        elapsed = time.time() - tick_start
        sleep_time = max(0.0, tick_interval - elapsed)
        if elapsed > 1.5 * tick_interval:
            log.warning(f"Tick overrun: {elapsed:.1f}s")

        # 7. Status every 10 ticks
        if tick % 10 == 0:
            n_findings = len(memory.get_findings())
            n_total = memory.count()
            log.info(f"  Status: {n_total} episodes, {n_findings} findings")

        time.sleep(sleep_time)

    # Shutdown
    memory.flush()
    log.info(f"Shutdown after {tick} ticks. {memory.count()} episodes stored.")


if __name__ == "__main__":
    main()
