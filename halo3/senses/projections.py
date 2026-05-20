"""SenseProjections — learned gating injection of sense signal into text tokens.

Architecture:
  audio_jax:  (8, 768)   -> audio_proj (no bias) -> (8, 2048) -> mean -> (2048,)
  vision_jax: (512,)     -> vision_proj (no bias) -> (2048,)
  sense_context = mean(audio_ctx, vision_ctx)      -> (2048,)
  gate = sigmoid(sense_gate(sense_context))         -> (2048,)
  output = text_tokens + gate * sense_context        -> (32, 2048)

Zero-input safety: audio_proj and vision_proj have use_bias=False.
  zero input -> zero output -> sense_context = 0 -> injection = 0.
"""
from __future__ import annotations
import logging
import os
import jax
import jax.numpy as jnp
import equinox as eqx

log = logging.getLogger(__name__)


class SenseProjections(eqx.Module):
    audio_proj: eqx.nn.Linear   # (audio_dim -> d_model), no bias
    vision_proj: eqx.nn.Linear  # (vision_dim -> d_model), no bias
    sense_gate: eqx.nn.Linear   # (d_model -> d_model), with bias

    def __init__(
        self,
        audio_dim: int,
        vision_dim: int,
        d_model: int,
        key: jnp.ndarray,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.audio_proj = eqx.nn.Linear(audio_dim, d_model, use_bias=False, key=k1)
        self.vision_proj = eqx.nn.Linear(vision_dim, d_model, use_bias=False, key=k2)
        self.sense_gate = eqx.nn.Linear(d_model, d_model, use_bias=True, key=k3)

    def inject(
        self,
        text_tokens: jnp.ndarray,  # (n_tokens, d_model)
        audio_jax: jnp.ndarray,    # (8, audio_dim) — zeros if unavailable
        vision_jax: jnp.ndarray,   # (vision_dim,)  — zeros if unavailable
    ) -> jnp.ndarray:              # (n_tokens, d_model)
        """Inject gated sense signal as additive residual into text tokens."""
        # Project each modality to d_model
        audio_emb = jax.vmap(self.audio_proj)(audio_jax)  # (8, d_model)
        vision_emb = self.vision_proj(vision_jax)           # (d_model,)

        # Aggregate: mean over temporal audio frames, average with vision
        audio_ctx = jnp.mean(audio_emb, axis=0)            # (d_model,)
        sense_context = (audio_ctx + vision_emb) / 2.0     # (d_model,)

        # Learned gate: which dimensions to let through
        gate = jax.nn.sigmoid(self.sense_gate(sense_context))  # (d_model,)

        # Additive residual: broadcast over token dimension
        return text_tokens + gate * sense_context[None, :]  # (n_tokens, d_model)


def save_sense_proj(proj: SenseProjections, path: str) -> None:
    """Save SenseProjections weights to {path}.eqx."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    eqx.tree_serialise_leaves(path + ".eqx", proj)
    log.info(f"SenseProjections saved to {path}.eqx")


def load_sense_proj(
    audio_dim: int,
    vision_dim: int,
    d_model: int,
    path: str,
) -> SenseProjections:
    """Load SenseProjections from {path}.eqx, or return fresh if not found."""
    template = SenseProjections(audio_dim, vision_dim, d_model, jax.random.PRNGKey(0))
    eqx_path = path + ".eqx"
    if not os.path.exists(eqx_path):
        log.info(f"No sense_proj checkpoint at {eqx_path} — initializing fresh.")
        return template
    try:
        proj = eqx.tree_deserialise_leaves(eqx_path, template)
        log.info(f"SenseProjections loaded from {eqx_path}")
        return proj
    except Exception as e:
        log.warning(f"SenseProjections load failed ({e}) — using fresh weights.")
        return template
