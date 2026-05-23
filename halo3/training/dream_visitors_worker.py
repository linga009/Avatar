"""GPU subprocess for Phase 5c — train FNO + contrastive on dream visitor pairs.

Spawned by main.py after dream_visitors.py generates pairs on CPU.
Loads model + sense_module, trains on enriched (audio, text) pairs,
saves updated sense_module. Exits → OS reclaims GPU.

This is where Avatar's spectral codes learn to resonate with speech —
not by transplanting Whisper's knowledge, but by training on enriched
dream content that Whisper and Kokoro generated while Avatar slept.
"""
from __future__ import annotations
import argparse
import gc
import json
import logging
import os


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("dream_visitors_worker")

    parser = argparse.ArgumentParser(description="Dream visitors GPU training")
    parser.add_argument("--pairs", required=True, help="Path to visitor_pairs.npz")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (without .eqx)")
    parser.add_argument("--sense-checkpoint", default="data/checkpoints/sense_module",
                        help="Sense module checkpoint (without .eqx)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    import optax

    xla_cache = os.path.abspath(os.path.join("data", "xla_cache"))
    os.makedirs(xla_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", xla_cache)

    backend = jax.default_backend()
    log.info(f"Dream visitors worker started — JAX backend: {backend}")

    # Load pairs
    if not os.path.exists(args.pairs):
        log.info("No visitor pairs file — skipping")
        return

    data = np.load(args.pairs, allow_pickle=True)
    audios = data["audios"]  # (N, 32000)
    texts = json.loads(str(data["texts"]))
    n_pairs = len(texts)
    log.info(f"Loaded {n_pairs} dream visitor pairs")

    if n_pairs == 0:
        return

    # Load model and sense module
    from halo3.config import Halo3Config
    from halo3.training.bootstrap import load_checkpoint
    from halo3.senses.sense_module import SenseModule, load_sense_module, save_sense_module
    from halo3.perception.native_embedder import NativeEmbedder
    from halo3.training.dream_replay import scale_by_clion

    cfg = Halo3Config()
    model = load_checkpoint(cfg, args.checkpoint)
    sense_module = load_sense_module(cfg, args.sense_checkpoint)

    # Embedder for text tokenization
    embedder = NativeEmbedder(cfg.d_model, n_tokens=cfg.n_tokens)

    # Optimizer for sense module only
    opt = optax.chain(
        optax.clip_by_global_norm(0.1),
        scale_by_clion(b1=0.9),
        optax.scale(-args.lr),
    )
    sm_params = eqx.filter(sense_module, eqx.is_array)
    opt_state = opt.init(sm_params)

    # Zero vision input (we're training audio-text alignment)
    zero_vision = jnp.zeros((224, 224, 3))

    # JIT-compiled training step
    @eqx.filter_jit
    def _visitor_step(sense_mod, opt_state_in, audio, text_emb, scale):
        def loss_fn(sm):
            # Run audio through FNO + VQ-VAE
            tokens_dummy = jnp.zeros((cfg.n_tokens, cfg.d_model))
            injected, info = sm.process_and_inject(tokens_dummy, audio, zero_vision)

            # Audio embedding: mean projected spectral code
            audio_emb = jnp.mean(
                jax.vmap(sm.spectral_proj)(info["audio_z_q"]), axis=0
            )

            # Contrastive loss: push audio embedding toward text embedding
            audio_n = audio_emb / (jnp.linalg.norm(audio_emb) + 1e-8)
            text_n = text_emb / (jnp.linalg.norm(text_emb) + 1e-8)
            contrastive = -jnp.sum(audio_n * text_n)

            # Commitment loss from VQ-VAE
            commitment = info["commitment_loss"]

            return (contrastive + 0.25 * commitment) * scale

        loss, grads = eqx.filter_value_and_grad(loss_fn)(sense_mod)
        updates, new_opt = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state_in,
            eqx.filter(sense_mod, eqx.is_array),
        )
        return eqx.apply_updates(sense_mod, updates), new_opt, loss

    log.info(f"Phase 5c: training FNO on {n_pairs} dream visitor pairs...")
    total_loss = 0.0
    completed = 0

    for i in range(n_pairs):
        audio = jnp.array(audios[i])
        text = texts[i]

        # Embed text
        text_tokens = embedder.texts_to_tokens([text], cfg.n_tokens)
        text_emb = jnp.mean(text_tokens, axis=(0, 1))  # (d_model,)

        try:
            sense_module, opt_state, loss = _visitor_step(
                sense_module, opt_state, audio, text_emb, jnp.float32(args.scale)
            )
            loss_val = float(loss)
            if loss_val == loss_val:  # NaN check
                total_loss += loss_val
                completed += 1
        except Exception as e:
            log.warning(f"  Visitor step {i} failed: {e}")

    avg_loss = total_loss / max(completed, 1)
    log.info(f"Phase 5c done: {completed}/{n_pairs} steps, avg_loss={avg_loss:.4f}")

    # Save updated sense module
    save_sense_module(sense_module, args.sense_checkpoint)
    log.info("Dream visitors: updated sense_module saved")

    # Write info for parent
    info = {"visitor_steps": completed, "visitor_loss": avg_loss}
    info_path = os.path.join(os.path.dirname(args.pairs), "visitor_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f)

    gc.collect()
    jax.clear_caches()


if __name__ == "__main__":
    main()
