"""BodyVocabulary — decode the body's learned representations into words.

The physics body (106M params) was trained on TinyStories via a weight-tied
LM head. Its backbone hidden states (h_out, d_model=2048) map back to tokens
through the embedding matrix transpose: logits = h @ embedding.T.

This module loads that embedding matrix and BPE tokenizer, then decodes
body tensors into semantic word lists — giving the PFC and chat interface
access to what the body is actually thinking, not just scalar metrics.
"""
from __future__ import annotations
import logging
import os

import numpy as np
import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)

# Sentencepiece special token IDs to filter
_SPECIAL_IDS = frozenset([0, 1, 2, 3])  # pad, unk, bos, eos
_MIN_WORD_LEN = 3
_TOP_K = 15


class BodyVocabulary:
    """Decode body tensors into semantic word lists via the trained LM head."""

    def __init__(self, lm_head_path: str, tokenizer_path: str) -> None:
        self._embedding = None
        self._tokenizer = None
        self._enabled = False

        # Cached decode results
        self._current_thoughts: list[str] = []
        self._recent_thoughts: list[str] = []
        self._deep_knowledge: list[str] = []
        self._expected_words: list[str] = []
        self._actual_words: list[str] = []

        if not os.path.exists(lm_head_path) or not os.path.exists(tokenizer_path):
            log.warning(
                "BodyVocabulary disabled — missing LM head or tokenizer "
                f"(looked for {lm_head_path}, {tokenizer_path})"
            )
            return

        try:
            self._load(lm_head_path, tokenizer_path)
        except Exception as e:
            log.warning(f"BodyVocabulary disabled — load failed: {e}")

    def _load(self, lm_head_path: str, tokenizer_path: str) -> None:
        import equinox as eqx
        import sentencepiece as spm
        from halo3.lm_head import LanguageModelHead
        from halo3.config import Halo3Config

        cfg = Halo3Config()
        lm_head = LanguageModelHead(cfg, jax.random.PRNGKey(0))
        lm_head = eqx.tree_deserialise_leaves(lm_head_path, lm_head)
        # Cast to float32 — bfloat16 checkpoints corrupt threshold comparison
        self._embedding = jnp.asarray(lm_head.embedding, dtype=jnp.float32)

        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        self._tokenizer = sp

        self._enabled = True
        log.info(
            f"BodyVocabulary enabled — embedding {self._embedding.shape}, "
            f"vocab {sp.GetPieceSize()}"
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _decode(self, h: jnp.ndarray, top_k: int = _TOP_K) -> list[str]:
        """Decode tensor(s) into a list of semantic words.

        Args:
            h: (N, d_model) or (d_model,) tensor to decode
            top_k: maximum number of words to return

        Returns:
            List of decoded words, filtered and sorted by score.
        """
        if not self._enabled:
            return []

        # Mean pool if multiple vectors
        if h.ndim == 2:
            h_mean = jnp.mean(h, axis=0)
        else:
            h_mean = h

        # Project to vocab logits via weight-tied embedding
        logits = h_mean @ self._embedding.T  # (vocab_size,)

        # Softmax for threshold filtering
        probs = jax.nn.softmax(logits)
        uniform_threshold = 1.0 / self._embedding.shape[0]

        # Top-k token IDs (oversample for filtering)
        top_indices = jnp.argsort(logits)[-top_k * 2:][::-1]

        # Materialize to numpy once — avoids per-element device-to-host transfers
        top_ids = np.asarray(top_indices)
        top_probs = np.asarray(probs[top_indices])

        words = []
        for i, idx_int in enumerate(top_ids.tolist()):
            if idx_int in _SPECIAL_IDS:
                continue
            if top_probs[i] < uniform_threshold:
                break  # sorted by score, rest will be below too

            piece = self._tokenizer.IdToPiece(idx_int)
            # Clean sentencepiece formatting
            word = piece.replace("\u2581", "").strip()
            if len(word) < _MIN_WORD_LEN:
                continue
            if word not in words:  # dedup
                words.append(word)
            if len(words) >= top_k:
                break

        return words

    def tick(
        self,
        h_out: jnp.ndarray,
        cache: jnp.ndarray,
        n_cached: int,
        island: jnp.ndarray,
        surprise: float,
        tick_num: int,
    ) -> None:
        """Decode body state on tiered schedule.

        Args:
            h_out: (n_tokens, d_model) — this tick's backbone output
            cache: (max_cache, d_model) — page memory ring buffer
            n_cached: number of valid entries in cache
            island: (island_size, d_model) — compressed past experience
            surprise: self_surprise from introspection (0-1)
            tick_num: current tick number
        """
        if not self._enabled:
            return

        # Always: decode live thoughts
        self._current_thoughts = self._decode(h_out)

        # Every 5 ticks: decode recent thinking
        if tick_num % 5 == 0 and n_cached > 0:
            max_cache = cache.shape[0]
            n_valid = min(n_cached, max_cache)
            self._recent_thoughts = self._decode(cache[:n_valid])

        # Every 50 ticks: decode deep knowledge from islands
        if tick_num % 50 == 0:
            self._deep_knowledge = self._decode(island)

        # On surprise spike: decode what was expected vs what arrived
        if surprise <= 0.3:
            self._expected_words = []
            self._actual_words = []
        elif n_cached > 0:
            max_cache = cache.shape[0]
            n_tokens = h_out.shape[0]
            write_ptr = n_cached % max_cache
            indices = (write_ptr - jnp.arange(1, n_tokens + 1)) % max_cache
            h_expected = cache[indices]
            self._expected_words = self._decode(h_expected)
            self._actual_words = self._decode(h_out)

    def get_body_words(self) -> dict:
        """Return snapshot of all decoded body vocabulary.

        Thread-safe: lists are replaced atomically (Python GIL).
        """
        return {
            "thoughts": list(self._current_thoughts),
            "recent": list(self._recent_thoughts),
            "knowledge": list(self._deep_knowledge),
            "expected": list(self._expected_words),
            "actual": list(self._actual_words),
        }
