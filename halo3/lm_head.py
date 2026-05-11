"""Language Model Head — gives the organism a voice.

Adds token embedding + LM head to the physics backbone.
Weight-tied: embedding and output projection share the same matrix.

Architecture:
  token IDs → Embedding (vocab × d_model) → MERA backbone → LM head → next token
  The LM head reuses the embedding weight matrix (weight tying).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class LanguageModelHead(eqx.Module):
    """Token embedding + tied output projection."""
    embedding: jnp.ndarray   # (vocab_size, d_model) — shared with LM head
    vocab_size: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray) -> None:
        self.vocab_size = cfg.vocab_size
        self.d_model = cfg.d_model
        # Xavier initialization
        scale = 1.0 / (cfg.d_model ** 0.5)
        self.embedding = jax.random.normal(key, (cfg.vocab_size, cfg.d_model)) * scale

    def embed(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """Convert token IDs to embeddings.

        Args: token_ids (seq_len,) int32
        Returns: (seq_len, d_model) float32
        """
        return self.embedding[token_ids]

    def project(self, h: jnp.ndarray) -> jnp.ndarray:
        """Project backbone output to vocab logits (weight-tied).

        Args: h (seq_len, d_model)
        Returns: (seq_len, vocab_size) logits
        """
        return h @ self.embedding.T  # weight tying


def lm_loss(lm_head, backbone, token_ids, lorentz_embed, key):
    """Next-token prediction loss through the physics backbone.

    Args:
        lm_head: LanguageModelHead
        backbone: Halo3Backbone
        token_ids: (seq_len,) int32 token IDs
        lorentz_embed: LorentzEmbedding
        key: PRNG key

    Returns:
        (loss, n_correct) — cross-entropy loss and number of correct predictions
    """
    seq_len = token_ids.shape[0]

    # Input: all tokens except last
    input_ids = token_ids[:-1]   # (seq_len-1,)
    target_ids = token_ids[1:]   # (seq_len-1,)

    # Embed tokens
    h = lm_head.embed(input_ids)  # (seq_len-1, d_model)

    # Get Lorentz coordinates for attention
    x, z = lorentz_embed(h)

    # Run through physics backbone
    h_out = backbone(h, x, z)  # (seq_len-1, d_model)

    # Project to vocab
    logits = lm_head.project(h_out)  # (seq_len-1, vocab_size)

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = log_probs[jnp.arange(seq_len - 1), target_ids]
    loss = -jnp.mean(target_log_probs)

    # Accuracy
    predictions = jnp.argmax(logits, axis=-1)
    n_correct = jnp.sum(predictions == target_ids)

    return loss, n_correct
