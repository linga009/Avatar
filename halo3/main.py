"""HoloBiont — A Living Research Organism.

Not a monitor. Not an engine. An organism that inhabits a physics body,
feels curiosity and boredom, builds a narrative identity, and dreams.

Usage:
    python -m halo3.main
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
    log.info("  HoloBiont 3.0 — A Living Research Organism")
    log.info("  Bohmian Holomovement Engine + Psyche")
    log.info("=" * 60)

    # --- Config ---
    from halo3.config import Halo3Config

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

    # --- Topics ---
    topics_path = os.path.join(os.path.dirname(__file__), "topics.yaml")
    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topics_cfg = yaml.safe_load(f)
    else:
        topics_cfg = {"seed_topics": ["artificial intelligence research"]}

    seed_topics = topics_cfg.get("seed_topics", ["AI research"])
    max_results = topics_cfg.get("max_results_per_query", 5)
    base_tick = topics_cfg.get("tick_interval", 60)

    # --- Model ---
    from halo3.model import Halo3Model, halo3_step
    from halo3.training.bootstrap import load_checkpoint
    from halo3.kuramoto import order_parameter

    checkpoint_path = "data/checkpoints/halo3"
    try:
        model = load_checkpoint(cfg, checkpoint_path)
        log.info(f"Loaded checkpoint from {checkpoint_path}.eqx")
    except Exception:
        log.info("No checkpoint — initializing fresh model")
        model = Halo3Model(cfg, jax.random.PRNGKey(cfg.seed))

    carry = model.init_carry(jax.random.PRNGKey(cfg.seed))
    key = jax.random.PRNGKey(cfg.seed + 1)

    # --- Components ---
    from halo3.perception.pipeline import PerceptionPipeline
    from halo3.memory.schema import Episode
    from halo3.memory.episode_store import EpisodeStore
    from halo3.psyche.organism import Organism

    perception = PerceptionPipeline(cfg.d_model, cfg.n_tokens)
    memory = EpisodeStore()
    organism = Organism(seed_topics)

    log.info(f"Organism awakening. {organism.self_model.identity_statement}")
    log.info(f"Watching: {seed_topics}")
    log.info("-" * 60)

    # --- Signals ---
    shutdown = False

    def _handle_signal(sig, frame):
        nonlocal shutdown
        log.info(f"Signal {sig} — organism entering shutdown")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- Heartbeat ---
    tick = 0
    current_query = seed_topics[0]
    prev_fe = None

    while not shutdown:
        tick += 1
        tick_start = time.time()

        # Circadian: tired organisms think slower
        tick_interval = organism.clock.modulate_tick_interval(base_tick, organism.drives.fatigue)

        # 1. PERCEIVE
        try:
            tokens, texts = perception.perceive(current_query, max_results)
        except Exception as e:
            log.warning(f"Perception failed: {e}")
            tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
            texts = []

        # 2. PHYSICS (the body)
        key, sk = jax.random.split(key)
        try:
            carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)
        except Exception as e:
            log.error(f"halo3_step failed: {e}")
            time.sleep(tick_interval)
            continue

        # 3. MEASURE (extract physics outputs)
        r = order_parameter(carry.kuramoto.theta)
        r_mean = float(jnp.mean(r))

        # Free energy proxy: reconstruction error
        fe = float(jnp.mean((q_final - q_data) ** 2))
        fe_delta = (fe - prev_fe) if prev_fe is not None else 0.0
        prev_fe = fe

        # 4. FEEL (the psyche)
        psyche_output = organism.tick(r_mean, fe_delta, texts, current_query)
        emotion = psyche_output["emotion"]
        finding = psyche_output["finding"]
        current_query = psyche_output["next_query"]

        # 5. REMEMBER
        query_embed = perception.embed_query(current_query)
        episode = Episode(
            query=current_query,
            order_param=r_mean,
            mode=emotion,  # emotion IS the mode now
            finding=finding,
            query_embed=query_embed,
            free_energy_delta=fe_delta,
        )
        memory.add(episode)

        # 6. EXPRESS (log the lived experience)
        r_bar = "█" * int(r_mean * 20) + "░" * (20 - int(r_mean * 20))
        log.info(
            f"Tick {tick:4d} | r=[{r_bar}] {r_mean:.3f} | "
            f"{psyche_output['log_line']}"
        )
        log.info(
            f"         | q=\"{current_query[:55]}\" | FE_Δ={fe_delta:+.4f}"
        )
        if finding:
            log.info(f"         → DISCOVERY: {finding[:90]}")

        # 7. Status every 10 ticks
        if tick % 10 == 0:
            log.info(f"  ◆ {organism.status()}")
            log.info(f"  ◆ Episodes: {memory.count()} | Findings: {len(memory.get_findings())}")

        # 8. DREAM (when the body needs it)
        if psyche_output["needs_dream"]:
            log.info("  ☽ Entering dream state — fine-tuning prefrontal cortex...")
            organism.dream(memory=memory)
            log.info(f"  ☽ Awoke. {organism.self_model.identity_statement}")

        # 9. SLEEP (the body rests between ticks)
        elapsed = time.time() - tick_start
        sleep_time = max(0.0, tick_interval - elapsed)
        if elapsed > 1.5 * tick_interval:
            log.warning(f"Tick overrun: {elapsed:.1f}s (interval={tick_interval:.0f}s)")
        time.sleep(sleep_time)

    # --- Shutdown ---
    memory.flush()
    organism.self_model.save()
    log.info(f"Organism resting after {tick} ticks.")
    log.info(f"Final identity: {organism.self_model.identity_statement}")
    log.info(f"Narrative: {len(organism.self_model.narrative)} entries")
    log.info(f"Episodes stored: {memory.count()}")


if __name__ == "__main__":
    main()
