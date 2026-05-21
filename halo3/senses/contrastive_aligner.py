"""ContrastiveAligner — InfoNCE loss for speech-text alignment.

Maintains a ring buffer of recent text embeddings as negatives.
Computes InfoNCE contrastive loss between audio and text embeddings.
Tracks codebook utilization for maturation gate.
"""
from __future__ import annotations
import logging
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


class ContrastiveAligner:
    """InfoNCE contrastive alignment for speech-text binding."""

    def __init__(self, embed_dim: int, buffer_size: int = 16,
                 tau: float = 0.07) -> None:
        self._embed_dim = embed_dim
        self._buffer_size = buffer_size
        self._tau = tau
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._usage_history: deque[np.ndarray] = deque(maxlen=200)
        self._matured = False

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    @property
    def matured(self) -> bool:
        return self._matured

    def push_text_emb(self, text_emb) -> None:
        self._buffer.append(np.array(text_emb))

    def push_indices(self, indices) -> None:
        self._usage_history.append(np.array(indices, dtype=np.int32))

    def compute_loss(self, audio_emb: jnp.ndarray,
                     text_emb: jnp.ndarray) -> jnp.ndarray:
        if self._matured or len(self._buffer) < 2:
            return jnp.float32(0.0)
        audio_n = audio_emb / (jnp.linalg.norm(audio_emb) + 1e-8)
        text_n = text_emb / (jnp.linalg.norm(text_emb) + 1e-8)
        neg_stack = jnp.array(np.stack(list(self._buffer)))
        neg_n = neg_stack / (jnp.linalg.norm(neg_stack, axis=-1, keepdims=True) + 1e-8)
        sim_pos = jnp.sum(audio_n * text_n) / self._tau
        sim_neg = (audio_n @ neg_n.T) / self._tau
        logits = jnp.concatenate([sim_pos[None], sim_neg])
        loss = -sim_pos + jax.nn.logsumexp(logits)
        return loss

    def compute_utilization(self, codebook_size: int, window: int = 100) -> float:
        if not self._usage_history:
            return 0.0
        recent = list(self._usage_history)[-window:]
        all_indices = np.concatenate(recent)
        unique = len(np.unique(all_indices))
        return unique / codebook_size

    def check_maturation(self, codebook_size: int,
                         threshold: float = 0.75) -> bool:
        util = self.compute_utilization(codebook_size)
        if util >= threshold:
            self._matured = True
            log.info(
                f"Phoneme perception matured -- contrastive scaffold dropped "
                f"(utilization={util:.2f} >= {threshold:.2f})")
        return self._matured
