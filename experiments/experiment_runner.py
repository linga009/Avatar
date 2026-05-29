"""Ablation experiment runner — config-driven tick loop with CSV logging.

Wraps the main.py tick logic, swaps components based on ExperimentConfig
flags, and logs per-tick metrics to CSV for analysis.

Usage:
    python -m experiments.experiment_runner full_avatar --ticks 200
    python -m experiments.experiment_runner --all --ticks 100
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

from experiments.configs import ExperimentConfig, EXPERIMENTS, get_experiment_config
from experiments.metrics_logger import MetricsLogger

log = logging.getLogger(__name__)


def run_experiment(exp: ExperimentConfig) -> Path:
    """Run a single ablation experiment and return path to the CSV results.

    Args:
        exp: experiment configuration with component flags

    Returns:
        Path to the generated CSV file
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    log.info("=" * 60)
    log.info(f"  Experiment: {exp.name}")
    log.info(f"  Description: {exp.description}")
    log.info(f"  Ticks: {exp.n_ticks}")
    log.info(f"  Flags: cop={not exp.disable_cop} senses={not exp.disable_senses} "
             f"dreams={not exp.disable_dreams} Q={not exp.disable_quantum_potential} "
             f"transformer={exp.is_transformer_baseline}")
    log.info("=" * 60)

    # --- Transformer baseline: delegate to dedicated module ---
    if exp.is_transformer_baseline:
        log.info("Transformer baseline — delegating to transformer_baseline.py")
        from experiments.transformer_baseline import run_transformer_baseline
        return run_transformer_baseline(exp)

    # --- Config ---
    from halo3.config import Halo3Config

    backend = jax.default_backend()
    log.info(f"JAX backend: {backend}, devices: {jax.devices()}")

    if backend in ("gpu", "cuda"):
        cfg = Halo3Config(disable_quantum_potential=exp.disable_quantum_potential)
    else:
        cfg = Halo3Config(
            d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
            d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
            n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
            mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
            meta_n_hidden=4, meta_n_actions=2, meta_k=3,
            max_cache=8, island_size=4,
            disable_quantum_potential=exp.disable_quantum_potential,
        )

    # --- Topics ---
    topics_path = os.path.join(os.path.dirname(__file__), "..", "halo3", "topics.yaml")
    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topics_cfg = yaml.safe_load(f)
    else:
        topics_cfg = {"seed_topics": ["artificial intelligence research"]}

    seed_topics = topics_cfg.get("seed_topics", ["AI research"])
    max_results = topics_cfg.get("max_results_per_query", 5)

    # --- Model ---
    from halo3.model import Halo3Model, halo3_step
    from halo3.training.bootstrap import load_checkpoint
    from halo3.kuramoto import order_parameter, dual_order_parameters

    checkpoint_path = "data/checkpoints/halo3"
    try:
        model = load_checkpoint(cfg, checkpoint_path)
        log.info(f"Loaded checkpoint from {checkpoint_path}.eqx")
    except Exception:
        log.info("No checkpoint — creating fresh model")
        model = Halo3Model(cfg, jax.random.PRNGKey(cfg.seed))

    carry = model.init_carry(jax.random.PRNGKey(cfg.seed))
    key = jax.random.PRNGKey(cfg.seed + 1)

    # --- Components ---
    from halo3.perception.pipeline import PerceptionPipeline
    from halo3.psyche.organism import Organism
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import load_sense_module
    from halo3.senses.sensory_stats import SensoryStatistics

    perception = PerceptionPipeline(cfg.d_model, cfg.n_tokens)
    organism = Organism(seed_topics, cfg=cfg)
    predictor = PredictiveProcessor(lr=1e-5)

    # --- Senses ---
    sense_module = load_sense_module(cfg, path="data/checkpoints/sense_module")
    sensory_stats = SensoryStatistics(
        audio_tokens=cfg.n_audio_tokens,
        vision_tokens=cfg.n_vision_tokens,
        codebook_size=cfg.codebook_size_audio,
    )
    sensory_stats.load("data/sensory_stats.json")
    _sense_zero_audio = jnp.zeros((32000,))
    _sense_zero_vision = jnp.zeros((224, 224, 3))

    # --- Component swaps based on experiment flags ---

    # disable_cop: replace organism's COP engine with NullCOP
    if exp.disable_cop:
        from experiments.no_cop import NullCOP
        organism.cop = NullCOP(cfg)
        log.info("COP DISABLED — using NullCOP (fixed chi/tau/unity, no SOC)")

    # disable_senses: we'll force zero arrays every tick (handled in loop)
    if exp.disable_senses:
        log.info("SENSES DISABLED — zero sensory injection every tick")

    # disable_dreams: we cap fatigue so needs_dream never fires (handled in loop)
    if exp.disable_dreams:
        log.info("DREAMS DISABLED — fatigue capped at 0.64")

    # disable_quantum_potential: already handled via cfg flag
    if exp.disable_quantum_potential:
        log.info("QUANTUM POTENTIAL DISABLED — pure Kuramoto without anti-bunching")

    # --- Metrics logger ---
    csv_path = Path("data/experiments") / f"{exp.name}.csv"
    logger = MetricsLogger(csv_path)

    # --- Tick loop ---
    current_query = seed_topics[0]
    prev_fe = None
    unique_topics: set[str] = set()

    log.info(f"Starting {exp.n_ticks} ticks...")
    t_start = time.time()

    for tick in range(1, exp.n_ticks + 1):
        tick_start = time.time()

        # -- Dreams suppression: cap fatigue before it can trigger --
        if exp.disable_dreams:
            if organism.drives.fatigue > 0.64:
                organism.drives.fatigue = 0.64

        # 1. PERCEIVE — text
        try:
            key, subkey = jax.random.split(key)
            tokens, texts = perception.perceive(
                current_query, max_results, model=model, carry=carry, key=subkey)
        except Exception as e:
            log.warning(f"Perception failed: {e}")
            tokens = jnp.zeros((cfg.n_tokens, cfg.d_model))
            texts = []

        # 2. SENSE — spectral FNO perception
        if exp.disable_senses:
            audio_raw = _sense_zero_audio
            vision_raw = _sense_zero_vision
        else:
            audio_raw = _sense_zero_audio
            vision_raw = _sense_zero_vision

        # Inject sense signal into text tokens
        tokens, sense_info = sense_module.process_and_inject(tokens, audio_raw, vision_raw)

        # Update sensory statistics
        sensory_stats.update(sense_info["audio_indices"], sense_info["vision_indices"])

        # 3. PHYSICS — the body processes input
        key, sk = jax.random.split(key)
        try:
            carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)
        except Exception as e:
            log.error(f"halo3_step failed: {e}")
            continue

        # 4. LEARN — the body adapts
        key, lk = jax.random.split(key)
        try:
            model, sense_module, pred_loss, _learn_info = predictor.learn_from_error(
                model, sense_module, carry, tokens, audio_raw, vision_raw, q_data, lk,
            )
            pred_error = pred_loss
        except Exception as e:
            log.warning(f"Body learning failed: {e}")
            pred_error = float(jnp.mean((q_final - q_data) ** 2))

        # 5. MEASURE — extract physics outputs
        r = order_parameter(carry.kuramoto.theta)
        r_mean = float(jnp.mean(r))
        _r_a, _r_c, _body_tension = dual_order_parameters(carry.kuramoto.theta)
        body_tension = float(_body_tension)

        # Free energy proxy
        fe = float(jnp.mean((q_final - q_data) ** 2))
        fe_delta = (fe - prev_fe) if prev_fe is not None else 0.0
        prev_fe = fe

        # 6. FEEL — the psyche
        combined_surprise = fe_delta + pred_error * 0.001

        psyche_output = organism.tick(
            r_mean, combined_surprise, texts, current_query,
            body_tension=body_tension,
            r_a=float(_r_a),
            r_c=float(_r_c),
            theta=carry.kuramoto.theta,
            K_aa=float(carry.kuramoto.coupling_aa),
            K_cc=float(carry.kuramoto.coupling_cc),
            K_cross=float(carry.kuramoto.coupling_cross),
        )

        emotion = psyche_output["emotion"]
        intensity = psyche_output["intensity"]
        finding = psyche_output["finding"]
        current_query = psyche_output["next_query"]

        # COP: set coupling K from SOC controller (block coupling)
        carry = carry._replace(
            kuramoto=carry.kuramoto._replace(
                coupling_aa=psyche_output["K_aa"],
                coupling_cc=psyche_output["K_cc"],
                coupling_cross=psyche_output["K_cross"],
            )
        )

        # Track topic diversity
        unique_topics.add(current_query)

        # 7. LOG — write CSV row
        logger.log(
            tick=tick,
            r_mean=round(r_mean, 6),
            fe_delta=round(fe_delta, 6),
            chi=round(psyche_output.get("chi", 0.0), 6),
            tau=round(psyche_output.get("tau", 0.0), 6),
            K=round(float(new_K), 6),
            unity=round(psyche_output.get("unity", 0.0), 6),
            emotion=emotion,
            intensity=round(intensity, 4),
            query=current_query[:80],
            prediction_error=round(pred_error, 6),
            discovery=finding[:120] if finding else "",
            topic_diversity=len(unique_topics),
        )

        # 8. EXPRESS — console log
        elapsed = time.time() - tick_start
        r_bar = "#" * int(max(0, min(1, r_mean)) * 20) + "." * (20 - int(max(0, min(1, r_mean)) * 20))
        log.info(
            f"[{exp.name}] Tick {tick:4d}/{exp.n_ticks} | "
            f"r=[{r_bar}] {r_mean:.3f} | {emotion:12s} i={intensity:.2f} | "
            f"K={new_K:.3f} chi={psyche_output.get('chi', 0):.2f} | "
            f"FE={fe_delta:+.2e} | {elapsed:.1f}s"
        )

        # 9. DREAM handling
        if psyche_output["needs_dream"]:
            if exp.disable_dreams:
                # Should not happen due to fatigue cap, but safety net
                log.info(f"[{exp.name}] Dream suppressed (ablation)")
                organism.drives.fatigue = 0.3
            else:
                # Reset drives without running full dream subprocess
                log.info(f"[{exp.name}] Dream triggered — resetting drives (no subprocess)")
                organism.drives.dream_reset()

    # --- Finalize ---
    logger.close()
    total_time = time.time() - t_start
    log.info("=" * 60)
    log.info(f"  Experiment '{exp.name}' complete")
    log.info(f"  {exp.n_ticks} ticks in {total_time:.1f}s ({total_time / max(1, exp.n_ticks):.2f}s/tick)")
    log.info(f"  Results: {csv_path}")
    log.info(f"  Topic diversity: {len(unique_topics)} unique queries")
    log.info("=" * 60)

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Avatar ablation experiments",
        prog="experiments.experiment_runner",
    )
    parser.add_argument(
        "name", nargs="?", default=None,
        help=f"Experiment name. Options: {list(EXPERIMENTS.keys())}",
    )
    parser.add_argument(
        "--ticks", type=int, default=None,
        help="Override number of ticks (default: use config value)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments sequentially",
    )
    args = parser.parse_args()

    if args.all:
        results = {}
        for name, exp_cfg in EXPERIMENTS.items():
            if args.ticks is not None:
                exp_cfg = ExperimentConfig(
                    name=exp_cfg.name,
                    n_ticks=args.ticks,
                    disable_cop=exp_cfg.disable_cop,
                    disable_senses=exp_cfg.disable_senses,
                    disable_dreams=exp_cfg.disable_dreams,
                    disable_quantum_potential=exp_cfg.disable_quantum_potential,
                    is_transformer_baseline=exp_cfg.is_transformer_baseline,
                    description=exp_cfg.description,
                )
            csv = run_experiment(exp_cfg)
            results[name] = csv
        print("\n=== ALL EXPERIMENTS COMPLETE ===")
        for name, csv in results.items():
            print(f"  {name}: {csv}")
        return

    if args.name is None:
        parser.error("Provide an experiment name or use --all")

    exp = get_experiment_config(args.name)
    if args.ticks is not None:
        exp = ExperimentConfig(
            name=exp.name,
            n_ticks=args.ticks,
            disable_cop=exp.disable_cop,
            disable_senses=exp.disable_senses,
            disable_dreams=exp.disable_dreams,
            disable_quantum_potential=exp.disable_quantum_potential,
            is_transformer_baseline=exp.is_transformer_baseline,
            description=exp.description,
        )
    csv_path = run_experiment(exp)
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
