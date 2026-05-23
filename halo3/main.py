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

import equinox as eqx
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
    from halo3.kuramoto import order_parameter, dual_order_parameters

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
    from halo3.senses.sense_buffer import SenseBuffer
    from halo3.senses.sense_module import SenseModule, load_sense_module, save_sense_module, delete_decoders
    from halo3.senses.sensory_stats import SensoryStatistics
    from halo3.senses.tts_narration import TTSNarrator, extract_narration_text
    from halo3.senses.contrastive_aligner import ContrastiveAligner

    perception = PerceptionPipeline(cfg.d_model, cfg.n_tokens)
    memory = EpisodeStore()
    organism = Organism(seed_topics)
    predictor = PredictiveProcessor(lr=1e-5)

    # --- Chat server (talk to the whole organism) ---
    from halo3.chat_server import start_chat_server, update_live_state
    start_chat_server(port=8420)

    # --- Senses (spectral FNO + VQ-VAE) ---
    sense_buffer = SenseBuffer(data_dir="data", stale_threshold_secs=30.0)
    sense_module = load_sense_module(cfg, path="data/checkpoints/sense_module")
    sensory_stats = SensoryStatistics(
        audio_tokens=cfg.n_audio_tokens, vision_tokens=cfg.n_vision_tokens,
        codebook_size=cfg.codebook_size_audio)
    sensory_stats.load("data/sensory_stats.json")
    _sense_zero_audio = jnp.zeros((32000,))
    _sense_zero_vision = jnp.zeros((224, 224, 3))
    _first_dream_done = not sense_module.has_decoders
    log.info(f"Senses: FNO spectral cortex ({'critical period' if sense_module.has_decoders else 'mature'})")

    # --- TTS self-narration + contrastive alignment (v3.8) ---
    tts = TTSNarrator(mode=cfg.tts_mode, sample_rate=16000, duration_samples=32000)
    contrastive_aligner = ContrastiveAligner(
        embed_dim=cfg.d_model, buffer_size=16, tau=cfg.contrastive_tau)
    log.info(f"TTS: {cfg.tts_mode} ({'ON' if tts.available else 'OFF'}) | "
             f"Contrastive: tau={cfg.contrastive_tau}, weight={cfg.contrastive_weight}")

    log.info(f"Organism awakening. {organism.self_model.identity_statement}")
    log.info(f"Watching: {seed_topics}")
    log.info(f"Predictive processing: ON — body learns every tick")
    log.info(f"Chat: http://localhost:8420 — talk to the living organism")
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

        # 1. PERCEIVE — text
        try:
            tokens, texts = perception.perceive(current_query, max_results)
        except Exception as e:
            log.warning(f"Perception failed: {e}")
            tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
            texts = []

        perception_failed = len(texts) == 0

        # 1b. SENSE — spectral FNO perception
        raw_data = sense_buffer.get_raw_arrays()
        audio_raw = jnp.array(raw_data.audio_np) if raw_data.audio_np is not None else _sense_zero_audio
        vision_raw = jnp.array(raw_data.vision_np) if raw_data.vision_np is not None else _sense_zero_vision
        sense_label = "[ ][ ]"
        if raw_data.audio_np is not None:
            sense_label = "[A][ ]"
        if raw_data.vision_np is not None:
            sense_label = sense_label.replace("[ ]", "[V]", 1)

        # TTS self-narration mixing (v3.8)
        import numpy as _np
        text_paired = False
        if tts.available and not contrastive_aligner.matured:
            use_tts = (raw_data.audio_np is None) or (tick % cfg.tts_every_n == 0)
            if use_tts and texts:
                narration_text = extract_narration_text(texts, max_words=20)
                tts_audio = tts.narrate(narration_text)
                if tts_audio is not None and _np.any(tts_audio != 0):
                    audio_raw = jnp.array(tts_audio)
                    text_paired = True
                    sense_label = sense_label[:3] + "[T]"

        # Inject sense signal into text tokens (shape stays (n_tokens, d_model))
        tokens, sense_info = sense_module.process_and_inject(tokens, audio_raw, vision_raw)

        # Update sensory statistics for PFC
        sensory_stats.update(sense_info["audio_indices"], sense_info["vision_indices"])

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
            model, sense_module, pred_loss, _learn_info = predictor.learn_from_error(
                model, sense_module, carry, tokens, audio_raw, vision_raw, q_data, lk,
                contrastive_aligner=contrastive_aligner,
                text_paired=text_paired,
                contrastive_weight=cfg.contrastive_weight,
            )

            # EMA codebook update (outside gradient)
            sense_module = eqx.tree_at(
                lambda m: m.audio_codebook,
                sense_module,
                sense_module.audio_codebook.ema_update(
                    _learn_info["audio_z_e"], _learn_info["audio_indices"],
                    decay=cfg.codebook_ema_decay))
            sense_module = eqx.tree_at(
                lambda m: m.vision_codebook,
                sense_module,
                sense_module.vision_codebook.ema_update(
                    _learn_info["vision_z_e"], _learn_info["vision_indices"],
                    decay=cfg.codebook_ema_decay))

            # Dead code revival
            if tick % cfg.dead_code_threshold == 0:
                key, dk = jax.random.split(key)
                audio_usage = jnp.array(sensory_stats.audio_usage_counts)
                vision_usage = jnp.array(sensory_stats.vision_usage_counts)
                sense_module = eqx.tree_at(
                    lambda m: m.audio_codebook,
                    sense_module,
                    sense_module.audio_codebook.revive_dead_codes(
                        audio_usage, _learn_info["audio_z_e"], cfg.dead_code_threshold, dk))
                key, dk2 = jax.random.split(key)
                sense_module = eqx.tree_at(
                    lambda m: m.vision_codebook,
                    sense_module,
                    sense_module.vision_codebook.revive_dead_codes(
                        vision_usage, _learn_info["vision_z_e"], cfg.dead_code_threshold, dk2))

            # Contrastive: push text embedding and track indices
            if text_paired:
                text_emb_mean = jnp.mean(tokens, axis=0)
                contrastive_aligner.push_text_emb(text_emb_mean)
            contrastive_aligner.push_indices(_learn_info["audio_indices"])

            # Register speech codes from TTS ticks
            if text_paired:
                new_codes = sensory_stats._speech_codes | set(int(x) for x in _learn_info["audio_indices"])
                sensory_stats.register_speech_codes(new_codes)

            pred_error = pred_loss
        except Exception as e:
            log.warning(f"Body learning failed: {e}")
            pred_error = float(jnp.mean((q_final - q_data) ** 2))
            pred_loss = pred_error

        # 6. MEASURE (extract physics outputs)
        r = order_parameter(carry.kuramoto.theta)
        r_mean = float(jnp.mean(r))
        _, _, _body_tension = dual_order_parameters(carry.kuramoto.theta)
        body_tension = float(_body_tension)

        # Free energy proxy: reconstruction error + prediction error
        fe = float(jnp.mean((q_final - q_data) ** 2))
        fe_delta = (fe - prev_fe) if prev_fe is not None else 0.0
        prev_fe = fe

        # 7. FEEL (the psyche — now informed by prediction error too)
        # Combine FE delta with prediction error for richer surprise signal
        combined_surprise = fe_delta + pred_error * 0.001  # scale pred_error
        # Compute carry norm for introspective monitoring
        try:
            carry_leaves = jax.tree_util.tree_leaves(carry)
            carry_norm = float(sum(jnp.sum(l**2) for l in carry_leaves if hasattr(l, 'shape') and l.size > 1) ** 0.5)
        except Exception:
            carry_norm = None
        # Compute sensory scalars for psyche integration
        _s_arousal = (sensory_stats.audio_flux + sensory_stats.vision_flux) / max(1, cfg.n_audio_tokens + cfg.n_vision_tokens)
        _s_novelty = (sensory_stats.audio_novelty + sensory_stats.vision_novelty) / 2.0
        _s_stability = min(sensory_stats.audio_stability, sensory_stats.vision_stability)
        _s_speech = sensory_stats.speech_detected
        _s_binding = sensory_stats.cross_modal_binding

        # Speech recognition: transcribe when speech detected (CPU, ~500ms)
        _heard_speech = ""
        if _s_speech and raw_data.audio_np is not None:
            from halo3.senses.speech_recognition import transcribe
            _heard_speech = transcribe(raw_data.audio_np)
            if _heard_speech:
                log.info(f"  Heard: \"{_heard_speech[:60]}\"")

        psyche_output = organism.tick(
            r_mean, combined_surprise, texts, current_query,
            carry_norm=carry_norm, body_tension=body_tension,
            sensory_arousal=_s_arousal,
            sensory_novelty=_s_novelty,
            sensory_stability=_s_stability,
            speech_detected=_s_speech,
            binding_familiarity=_s_binding,
            sensory_stats_line=sensory_stats.format_for_pfc(),
            heard_speech=_heard_speech,
        )
        emotion = psyche_output["emotion"]
        finding = psyche_output["finding"]
        current_query = psyche_output["next_query"]

        # Update live state for chat server
        update_live_state(
            tick=tick, r_mean=r_mean, fe_delta=fe_delta,
            pred_error=pred_error, current_query=current_query,
            texts=texts, organism=organism, memory=memory, predictor=predictor,
            sensory_stats_line=sensory_stats.format_for_pfc(),
        )

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
            mode=emotion,
            finding=finding,
            query_embed=query_embed,
            free_energy_delta=fe_delta,
            audio_codes=list(int(x) for x in sense_info["audio_indices"]),
            vision_codes=list(int(x) for x in sense_info["vision_indices"]),
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
            f"         | q=\"{current_query[:45]}\" | FE_Δ={fe_delta:+.2e} | ε={pred_error:.2e}{improving}{fail_marker} | {sense_label}"
        )
        if finding:
            log.info(f"         → DISCOVERY: {finding[:90]}")

        # 7. Status every 10 ticks
        if tick % 10 == 0:
            log.info(f"  ◆ {organism.status()}")
            log.info(f"  ◆ Episodes: {memory.count()} | Findings: {len(memory.get_findings())}")
            # Volatility surface snapshot
            vol_summary = organism.volatility.summary()
            if vol_summary:
                top3 = sorted(vol_summary.items(), key=lambda x: x[1]["V"], reverse=True)[:3]
                vol_str = " | ".join(f"{k}: σ={v['sigma']:.2f} V={v['V']:.4f}" for k, v in top3)
                log.info(f"  ◆ BS Valuation: {vol_str}")
            # Consciousness status
            ws = organism.workspace.summary()
            log.info(
                f"  ◆ Consciousness: "
                f"{'IGNITED' if ws['ignited'] else 'dark'} "
                f"(ratio={ws['consciousness_ratio']:.0%}) | "
                f"coherence={organism.temporal.temporal_coherence:.2f} | "
                f"meditations={organism.meditation.total_meditations} "
                f"insights={organism.meditation.total_insights}"
            )

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
            save_sense_module(sense_module, "data/checkpoints/sense_module")
            sensory_stats.save("data/sensory_stats.json")
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
                # Phases 1-3: body dream (replay + recombine + imagine)
                result = _sp.run(
                    [sys.executable, "-m", "halo3.training.dream_worker",
                     "--checkpoint", "data/checkpoints/pre_dream",
                     "--output", "data/checkpoints/halo3",
                     "--replay-steps", "10",
                     "--recombine-steps", "5",
                     "--imagine-steps", "5"],
                    timeout=3600,
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

            # GPU fully free — subprocess released everything on exit.
            # Phase 4: FineWeb batch in SEPARATE subprocess (isolated XLA state)
            try:
                log.info("  ☽ Phase 4: FineWeb dreaming on GPU (subprocess)...")
                fw_result = _sp.run(
                    [sys.executable, "-m", "halo3.training.dream_fineweb_worker",
                     "--checkpoint", "data/checkpoints/halo3",
                     "--output", "data/checkpoints/halo3",
                     "--steps", "10"],
                    timeout=1800,
                )
                if fw_result.returncode == 0:
                    fw_info_path = "data/checkpoints/fineweb_dream_info.json"
                    if os.path.exists(fw_info_path):
                        with open(fw_info_path) as f:
                            fw_info = _json.load(f)
                        log.info(f"  ☽ FineWeb dream done: {fw_info}")
                else:
                    log.warning(f"  ☽ FineWeb subprocess exited with code {fw_result.returncode}")
            except Exception as e:
                log.warning(f"  ☽ FineWeb dream failed (non-critical): {e}")

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
                    def _blend(old, new):
                        # Preserve integer types (page_mem counters, PRNG keys)
                        if hasattr(new, 'dtype') and not jnp.issubdtype(new.dtype, jnp.floating):
                            return new
                        return alpha * old + (1.0 - alpha) * new
                    carry = jax.tree_util.tree_map(_blend, _pre_dream_carry, fresh_carry)
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

            # End critical period after first dream
            if not _first_dream_done and sense_module.has_decoders:
                sense_module = delete_decoders(sense_module)
                _first_dream_done = True
                log.info("  ☽ Critical period ended -- sensory cortex matured")
                import gc; gc.collect()

            # Check contrastive maturation after dream (Phase B -> Phase C)
            if not contrastive_aligner.matured:
                contrastive_aligner.check_maturation(
                    cfg.codebook_size_audio, cfg.contrastive_maturation_threshold)

        # 9. SLEEP (the body rests between ticks)
        elapsed = time.time() - tick_start
        sleep_time = max(0.0, tick_interval - elapsed)
        if elapsed > 1.5 * tick_interval:
            log.warning(f"Tick overrun: {elapsed:.1f}s (interval={tick_interval:.0f}s)")
        time.sleep(sleep_time)

    # --- Shutdown ---
    save_sense_module(sense_module, "data/checkpoints/sense_module")
    sensory_stats.save("data/sensory_stats.json")
    memory.flush()
    organism.self_model.save()
    log.info(f"Organism resting after {tick} ticks.")
    log.info(f"Final identity: {organism.self_model.identity_statement}")
    log.info(f"Narrative: {len(organism.self_model.narrative)} entries")
    log.info(f"Episodes stored: {memory.count()}")


if __name__ == "__main__":
    main()
