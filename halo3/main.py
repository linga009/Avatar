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
import sys
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

    # --- XLA persistent compilation cache ---
    # First dream compiles XLA modules (~5 min each). Cache to disk so
    # subsequent dreams and container restarts reuse compiled binaries.
    xla_cache = os.path.join("data", "xla_cache")
    os.makedirs(xla_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", os.path.abspath(xla_cache))
    log.info(f"XLA compilation cache: {os.path.abspath(xla_cache)}")

    log.info("=" * 60)
    log.info("  Avatar 3.0 — A Living Research Organism")
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
        log.info("No organism checkpoint — birthing with LM-trained backbone")
        model = Halo3Model(cfg, jax.random.PRNGKey(cfg.seed))
        try:
            from halo3.training.lm_merge import load_lm_into_model
            model = load_lm_into_model(model, cfg)
            log.info("Born with LM-trained backbone (TinyStories 50K)")
        except FileNotFoundError as e:
            log.warning(f"LM weights not found, starting fully fresh: {e}")
        # Save so future restarts don't re-merge
        from halo3.training.bootstrap import save_checkpoint as _save
        _save(model, checkpoint_path)
        log.info(f"Saved initial organism to {checkpoint_path}.eqx")

    carry = model.init_carry(jax.random.PRNGKey(cfg.seed))
    key = jax.random.PRNGKey(cfg.seed + 1)

    # --- Components ---
    from halo3.perception.pipeline import PerceptionPipeline
    from halo3.memory.schema import Episode
    from halo3.memory.episode_store import EpisodeStore
    from halo3.psyche.organism import Organism

    from halo3.predictive import PredictiveProcessor

    perception = PerceptionPipeline(cfg.d_model, cfg.n_tokens)
    memory = EpisodeStore()
    organism = Organism(seed_topics)
    predictor = PredictiveProcessor(lr=1e-5)

    log.info(f"Organism awakening. {organism.self_model.identity_statement}")
    log.info(f"Watching: {seed_topics}")
    log.info(f"Predictive processing: ON — body learns every tick")
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
    _pre_dream_carry = None  # saved carry for warm-start after dreams

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

        perception_failed = len(texts) == 0

        # 2. PHYSICS (the body processes input)
        key, sk = jax.random.split(key)
        try:
            carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)
        except Exception as e:
            log.error(f"halo3_step failed: {e}")
            time.sleep(tick_interval)
            continue

        # 3. LEARN (the body adapts — weights change EVERY tick)
        key, lk = jax.random.split(key)
        try:
            model, pred_loss = predictor.learn_from_error(
                model, carry, tokens, q_data, lk
            )
            pred_error = pred_loss
        except Exception as e:
            log.warning(f"Body learning failed: {e}")
            pred_error = float(jnp.mean((q_final - q_data) ** 2))
            pred_loss = pred_error

        # 6. MEASURE (extract physics outputs)
        r = order_parameter(carry.kuramoto.theta)
        r_mean = float(jnp.mean(r))

        # Free energy proxy: reconstruction error + prediction error
        fe = float(jnp.mean((q_final - q_data) ** 2))
        fe_delta = (fe - prev_fe) if prev_fe is not None else 0.0
        prev_fe = fe

        # 7. FEEL (the psyche — now informed by prediction error too)
        # Combine FE delta with prediction error for richer surprise signal
        combined_surprise = fe_delta + pred_error * 0.001  # scale pred_error
        psyche_output = organism.tick(r_mean, combined_surprise, texts, current_query)
        emotion = psyche_output["emotion"]
        finding = psyche_output["finding"]
        current_query = psyche_output["next_query"]

        # 8. SATIATION — actively break Kuramoto sync when restless
        coupling_mod = psyche_output["coupling_mod"]
        if coupling_mod < 1.0:
            # Reduce coupling in the carry state to desynchronize
            old_K = carry.kuramoto.coupling
            new_K = old_K * coupling_mod
            carry = carry._replace(
                kuramoto=carry.kuramoto._replace(coupling=new_K)
            )

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
        r_safe = max(0.0, min(1.0, r_mean)) if (r_mean == r_mean) else 0.0
        r_bar = "█" * int(r_safe * 20) + "░" * (20 - int(r_safe * 20))
        log.info(
            f"Tick {tick:4d} | r=[{r_bar}] {r_mean:.3f} | "
            f"{psyche_output['log_line']}"
        )
        improving = "↑" if predictor.is_improving else "→"
        fail_marker = " ⊘ NO INPUT" if perception_failed else ""
        log.info(
            f"         | q=\"{current_query[:45]}\" | FE_Δ={fe_delta:+.2e} | ε={pred_error:.2e}{improving}{fail_marker}"
        )
        if finding:
            log.info(f"         → DISCOVERY: {finding[:90]}")

        # 7. Status every 10 ticks
        if tick % 10 == 0:
            log.info(f"  ◆ {organism.status()}")
            log.info(f"  ◆ Episodes: {memory.count()} | Findings: {len(memory.get_findings())}")

        # 8. DREAM (when the body needs it)
        if psyche_output["needs_dream"]:
            log.info("  ☽ Entering dream state — sequential body then mind...")

            # === PHASE 1: BODY DREAMS (GPU — isolated subprocess) ===
            # The subprocess gets the full GPU. When it exits, the OS
            # reclaims ALL GPU memory, XLA caches, and CPU allocations.
            log.info("  ☽ Phase 1: Body dreaming on GPU (subprocess)...")
            from halo3.training.bootstrap import save_checkpoint as _save_ckpt
            _save_ckpt(model, "data/checkpoints/pre_dream")
            _save_ckpt(model, "data/checkpoints/halo3")  # safety copy
            memory.flush()  # ensure episodes visible to subprocess

            # Save carry and predictor state for warm-start after dream
            _pre_dream_carry = carry
            predictor.save_state("data/predictor_state.npz")

            # Free GPU in parent BEFORE spawning subprocess
            del model
            jax.clear_caches()
            import gc; gc.collect()

            try:
                import subprocess as _sp
                import json as _json
                result = _sp.run(
                    [sys.executable, "-m", "halo3.training.dream_worker",
                     "--checkpoint", "data/checkpoints/pre_dream",
                     "--output", "data/checkpoints/halo3",
                     "--replay-steps", "10",
                     "--recombine-steps", "5",
                     "--imagine-steps", "5"],
                    timeout=3600,  # 1 hour max
                )
                if result.returncode == 0:
                    info_path = "data/checkpoints/halo3_dream_info.json"
                    if os.path.exists(info_path):
                        with open(info_path) as f:
                            dream_info = _json.load(f)
                        log.info(f"  ☽ Body dream done: {dream_info}")
                else:
                    log.warning(f"  ☽ Body dream subprocess exited with code {result.returncode}")
            except Exception as e:
                log.warning(f"  ☽ Body dream failed: {e}")

            # GPU is fully free — subprocess released everything on exit

            # === PHASE 2: MIND DREAMS (CPU) — LoRA fine-tune ===
            log.info("  ☽ Phase 2: Mind dreaming on CPU (LoRA fine-tune)...")
            organism.dream(memory=memory)

            # === PHASE 3: GEPA PROMPT EVOLUTION ===
            # Reflect on episode trajectories and evolve PFC prompt instructions.
            # Zero extra memory: uses Ollama (already running), no model loading.
            log.info("  ☽ Phase 3: GEPA prompt evolution...")
            try:
                from halo3.training.dream_gepa import dream_gepa, load_prompt_instructions
                all_episodes = memory.get_high_confidence(threshold=0.0)
                current_instrs = load_prompt_instructions()
                dream_gepa(all_episodes, current_instrs)
                # Tell the PFC to reload evolved instructions on next use
                organism.prefrontal.reload_instructions()
                log.info("  ☽ GEPA: prompt instructions evolved and reloaded")
            except Exception as e:
                log.warning(f"  ☽ GEPA failed (non-critical): {e}")

            # === RELOAD MODEL ===
            log.info("  ☽ Reloading physics body from checkpoint...")
            from halo3.training.bootstrap import load_checkpoint as _load_ckpt
            try:
                model = _load_ckpt(cfg, "data/checkpoints/halo3")
            except Exception:
                log.warning("  ☽ Checkpoint load failed — falling back to pre_dream")
                try:
                    model = _load_ckpt(cfg, "data/checkpoints/pre_dream")
                except Exception:
                    model = Halo3Model(cfg, jax.random.PRNGKey(cfg.seed))

            # Warm-start carry: blend pre-dream carry with fresh init
            # This preserves Kuramoto phase relationships learned before sleep
            fresh_carry = model.init_carry(jax.random.PRNGKey(cfg.seed))
            if _pre_dream_carry is not None:
                try:
                    alpha = 0.3  # 30% old carry, 70% fresh (dreamed body is new)
                    carry = jax.tree_util.tree_map(
                        lambda old, new: alpha * old + (1.0 - alpha) * new,
                        _pre_dream_carry, fresh_carry,
                    )
                    log.info("  ☽ Warm-started carry from pre-dream state (alpha=0.3)")
                except Exception:
                    carry = fresh_carry
                    log.info("  ☽ Carry warm-start failed, using fresh init")
                _pre_dream_carry = None
            else:
                carry = fresh_carry

            # Restore predictor optimizer state
            predictor.restore_state("data/predictor_state.npz")

            log.info(f"  ☽ Awoke. {organism.self_model.identity_statement}")
            log.info(f"  ☽ Prediction accuracy: {predictor.recent_prediction_accuracy:.4f}")

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
