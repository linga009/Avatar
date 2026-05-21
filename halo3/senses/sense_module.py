"""SenseModule — orchestrates FNO -> VQ-VAE -> Lorentz space injection.

Full sensory pipeline:
  1. Raw audio/vision -> FNO -> spectral features
  2. Spectral features -> VQ-VAE quantize -> discrete codes + quantized embeddings
  3. Quantized embeddings -> shared projection -> gated additive residual on text tokens

Also contains decoders for critical period (reconstruction loss).
"""
from __future__ import annotations
import logging
import os
import jax
import jax.numpy as jnp
import equinox as eqx

from halo3.senses.fno_audio import AudioFNO
from halo3.senses.fno_vision import VisionFNO
from halo3.senses.spectral_vqvae import SpectralCodebook

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoders — only alive during critical period
# ---------------------------------------------------------------------------

class AudioDecoder(eqx.Module):
    """Transposed 1D decoder for audio reconstruction (critical period only)."""
    expand: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    hidden_dim: int
    output_len: int

    def __init__(self, codebook_dim: int, hidden_dim: int, output_len: int,
                 *, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.expand = eqx.nn.Linear(codebook_dim, 2 * hidden_dim, use_bias=False, key=k1)
        self.proj_out = eqx.nn.Linear(hidden_dim, 1, use_bias=False, key=k2)
        self.hidden_dim = hidden_dim
        self.output_len = output_len

    def __call__(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """z_q: (n_tokens, codebook_dim) -> (output_len,) reconstructed waveform."""
        expanded = jax.vmap(self.expand)(z_q)                 # (n_tokens, 2*hidden)
        unfolded = expanded.reshape(-1, self.hidden_dim)      # (2*n_tokens, hidden)
        repeat_factor = self.output_len // unfolded.shape[0]
        upsampled = jnp.repeat(unfolded, repeat_factor, axis=0)
        upsampled = upsampled[: self.output_len]              # trim to exact length
        out = jax.vmap(self.proj_out)(upsampled)              # (output_len, 1)
        return out.squeeze(-1)


class VisionDecoder(eqx.Module):
    """Transposed 2D decoder for vision reconstruction (critical period only)."""
    expand: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    hidden_dim: int

    def __init__(self, codebook_dim: int, hidden_dim: int,
                 *, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.expand = eqx.nn.Linear(codebook_dim, 2 * hidden_dim, use_bias=False, key=k1)
        self.proj_out = eqx.nn.Linear(hidden_dim, 3, use_bias=False, key=k2)
        self.hidden_dim = hidden_dim

    def __call__(self, z_q: jnp.ndarray) -> jnp.ndarray:
        """z_q: (n_tokens, codebook_dim) -> (224, 224, 3) reconstructed image."""
        expanded = jax.vmap(self.expand)(z_q)                 # (n_tokens, 2*hidden)
        unfolded = expanded.reshape(-1, self.hidden_dim)      # (2*n_tokens, hidden)
        n_spatial = 224 * 224
        repeat_factor = n_spatial // unfolded.shape[0] + 1
        tiled = jnp.tile(unfolded, (repeat_factor, 1))[:n_spatial]
        spatial = tiled.reshape(224, 224, self.hidden_dim)
        out = jax.vmap(jax.vmap(self.proj_out))(spatial)      # (224, 224, 3)
        return out


# ---------------------------------------------------------------------------
# SenseModule — the orchestrator
# ---------------------------------------------------------------------------

class SenseModule(eqx.Module):
    """Full sensory pipeline: FNO -> VQ-VAE -> Lorentz injection."""
    audio_fno: AudioFNO
    vision_fno: VisionFNO
    audio_codebook: SpectralCodebook
    vision_codebook: SpectralCodebook
    spectral_proj: eqx.nn.Linear
    sense_gate: eqx.nn.Linear
    decoder_audio: AudioDecoder | None
    decoder_vision: VisionDecoder | None
    _has_decoders: bool

    def __init__(self, cfg, *, key: jnp.ndarray) -> None:
        keys = jax.random.split(key, 8)
        self.audio_fno = AudioFNO(
            hidden_dim=cfg.fno_hidden_dim, n_layers=cfg.fno_n_layers,
            modes=cfg.fno_audio_modes, n_tokens=cfg.n_audio_tokens,
            codebook_dim=cfg.codebook_dim, key=keys[0])
        self.vision_fno = VisionFNO(
            hidden_dim=cfg.fno_hidden_dim, n_layers=cfg.fno_n_layers,
            modes=cfg.fno_vision_modes, n_tokens=cfg.n_vision_tokens,
            codebook_dim=cfg.codebook_dim, key=keys[1])
        self.audio_codebook = SpectralCodebook(
            codebook_size=cfg.codebook_size, codebook_dim=cfg.codebook_dim, key=keys[2])
        self.vision_codebook = SpectralCodebook(
            codebook_size=cfg.codebook_size, codebook_dim=cfg.codebook_dim, key=keys[3])
        self.spectral_proj = eqx.nn.Linear(
            cfg.codebook_dim, cfg.d_model, use_bias=False, key=keys[4])
        self.sense_gate = eqx.nn.Linear(
            cfg.d_model, cfg.d_model, use_bias=True, key=keys[5])
        self.decoder_audio = AudioDecoder(
            codebook_dim=cfg.codebook_dim, hidden_dim=cfg.fno_hidden_dim,
            output_len=32000, key=keys[6])
        self.decoder_vision = VisionDecoder(
            codebook_dim=cfg.codebook_dim, hidden_dim=cfg.fno_hidden_dim,
            key=keys[7])
        self._has_decoders = True

    @property
    def has_decoders(self) -> bool:
        return self._has_decoders

    def process_and_inject(
        self,
        text_tokens: jnp.ndarray,
        audio_raw: jnp.ndarray,
        vision_raw: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        """Run full sensory pipeline and inject into text token stream.

        Args:
            text_tokens: (n_tokens, d_model) — current backbone embeddings
            audio_raw: (N,) — raw waveform (e.g. 32000 samples at 16kHz)
            vision_raw: (H, W, 3) — raw RGB image

        Returns:
            injected: (n_tokens, d_model) — text tokens with gated sense residual
            info: dict with indices, z_q, z_e, commitment_loss
        """
        # 1. FNO: raw -> spectral tokens
        audio_spectral = self.audio_fno(audio_raw)      # (n_audio_tokens, codebook_dim)
        vision_spectral = self.vision_fno(vision_raw)    # (n_vision_tokens, codebook_dim)

        # 2. VQ-VAE: quantize spectral tokens
        audio_z_q, audio_idx, audio_commit = self.audio_codebook.quantize(audio_spectral)
        vision_z_q, vision_idx, vision_commit = self.vision_codebook.quantize(vision_spectral)
        commitment_loss = audio_commit + vision_commit

        # 3. Project to d_model and compute gated injection
        audio_emb = jax.vmap(self.spectral_proj)(audio_z_q)    # (n_audio, d_model)
        vision_emb = jax.vmap(self.spectral_proj)(vision_z_q)  # (n_vision, d_model)
        sense_emb = jnp.concatenate([audio_emb, vision_emb], axis=0)
        sense_ctx = jnp.mean(sense_emb, axis=0)               # (d_model,)

        # Gated additive residual — sense_gate learns how much to inject
        gate = jax.nn.sigmoid(self.sense_gate(sense_ctx))      # (d_model,)
        injected = text_tokens + gate[None, :] * sense_ctx[None, :]

        info = {
            "audio_indices": audio_idx,
            "vision_indices": vision_idx,
            "commitment_loss": commitment_loss,
            "audio_z_q": audio_z_q,
            "vision_z_q": vision_z_q,
            "audio_z_e": audio_spectral,
            "vision_z_e": vision_spectral,
        }
        return injected, info

    def reconstruction_loss(
        self,
        audio_raw: jnp.ndarray,
        vision_raw: jnp.ndarray,
        info: dict,
    ) -> jnp.ndarray:
        """Compute MSE reconstruction loss (critical period training signal).

        Returns 0.0 if decoders have been deleted (post critical period).
        """
        if not self._has_decoders:
            return jnp.float32(0.0)
        audio_recon = self.decoder_audio(info["audio_z_q"])   # (32000,)
        vision_recon = self.decoder_vision(info["vision_z_q"])  # (224, 224, 3)
        audio_mse = jnp.mean((audio_recon - audio_raw) ** 2)
        vision_mse = jnp.mean((vision_recon - vision_raw) ** 2)
        return audio_mse + vision_mse


# ---------------------------------------------------------------------------
# Critical period lifecycle
# ---------------------------------------------------------------------------

def delete_decoders(sm: SenseModule) -> SenseModule:
    """Remove decoders from SenseModule (end of critical period).

    Uses eqx.tree_at to surgically replace decoder fields with None
    and flip the _has_decoders flag. This frees decoder parameters from
    memory while keeping the rest of the module intact.
    """
    sm = eqx.tree_at(
        lambda m: m.decoder_audio, sm, None,
        is_leaf=lambda x: x is sm.decoder_audio,
    )
    sm = eqx.tree_at(
        lambda m: m.decoder_vision, sm, None,
        is_leaf=lambda x: x is sm.decoder_vision,
    )
    sm = eqx.tree_at(lambda m: m._has_decoders, sm, False)
    return sm


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_sense_module(sm: SenseModule, path: str) -> None:
    """Save SenseModule to disk via equinox serialization."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    eqx_path = path + ".eqx"
    eqx.tree_serialise_leaves(eqx_path, sm)
    log.info(f"SenseModule saved to {eqx_path}")


def load_sense_module(cfg, path: str) -> SenseModule:
    """Load SenseModule from disk, falling back to fresh init on failure."""
    template = SenseModule(cfg, key=jax.random.PRNGKey(0))
    eqx_path = path + ".eqx"
    if not os.path.exists(eqx_path):
        log.info(f"No sense_module checkpoint at {eqx_path} -- initializing fresh.")
        return template
    try:
        sm = eqx.tree_deserialise_leaves(eqx_path, template)
        log.info(f"SenseModule loaded from {eqx_path}")
        return sm
    except Exception as e:
        log.warning(f"SenseModule load failed ({e}) -- using fresh weights.")
        return template
