"""Native Embedder — the organism perceives through its own trained body.

Replaces sentence-transformers with the organism's own LM head embedding.
The organism understands language through the same MERA backbone that
feels emotions and synchronizes patterns. No borrowed external model.

Falls back to sentence-transformers if no LM checkpoint exists.
"""
from __future__ import annotations
import logging
import os
import numpy as np
import jax
import jax.numpy as jnp

log = logging.getLogger(__name__)

LM_CHECKPOINT = "data/checkpoints/halo3_lm_lm.eqx"
TOKENIZER_MODEL = "data/tokenizer.model"


class NativeEmbedder:
    """Embeds text using the organism's own trained LM head.

    If the organism has been trained on TinyStories (LM checkpoint exists),
    uses its own embedding layer. Otherwise falls back to sentence-transformers.
    """

    def __init__(self, d_model: int, vocab_size: int = 8000, n_tokens: int = 32) -> None:
        self.d_model = d_model
        self.n_tokens = n_tokens
        self._native_ready = False
        self._embedding = None
        self._tokenizer = None
        self._fallback = None

        # Try to load native embedding
        if os.path.exists(LM_CHECKPOINT) and os.path.exists(TOKENIZER_MODEL):
            try:
                self._load_native(vocab_size)
            except Exception as e:
                log.warning(f"Native embedder failed to load: {e}")

        if not self._native_ready:
            log.info("Using sentence-transformers fallback (no LM checkpoint)")
            from halo3.perception.embedder import TextEmbedder
            self._fallback = TextEmbedder(d_model)

    def _load_native(self, vocab_size: int) -> None:
        """Load the organism's own trained embedding weights."""
        import sentencepiece as spm
        from halo3.config import Halo3Config
        from halo3.lm_head import LanguageModelHead
        import equinox as eqx

        # Load BPE tokenizer
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.Load(TOKENIZER_MODEL)

        # Load trained LM head (contains embedding matrix)
        cfg = Halo3Config()
        template = LanguageModelHead(cfg, jax.random.PRNGKey(0))
        lm_head = eqx.tree_deserialise_leaves(LM_CHECKPOINT, template)
        self._embedding = np.array(lm_head.embedding)  # (vocab, d_model)

        self._native_ready = True
        log.info(f"Native embedder loaded: {self._embedding.shape[0]} vocab, {self._embedding.shape[1]} dims")

    def texts_to_tokens(self, texts: list[str], n_tokens: int = None) -> jnp.ndarray:
        """Embed texts → (n_tokens, d_model) tensor."""
        n_tokens = n_tokens or self.n_tokens

        if self._native_ready:
            return self._embed_native(texts, n_tokens)
        else:
            return self._fallback.texts_to_tokens(texts, n_tokens)

    def _embed_native(self, texts: list[str], n_tokens: int) -> jnp.ndarray:
        """Embed using the organism's own vocabulary."""
        # Tokenize all texts into one sequence
        all_ids = []
        for text in texts:
            ids = self._tokenizer.Encode(text)
            all_ids.extend(ids)

        # Truncate/pad to n_tokens
        if len(all_ids) >= n_tokens:
            all_ids = all_ids[:n_tokens]
        else:
            all_ids.extend([0] * (n_tokens - len(all_ids)))

        # Look up embeddings
        token_ids = np.array(all_ids, dtype=np.int32)
        embedded = self._embedding[token_ids]  # (n_tokens, d_model)
        return jnp.array(embedded)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed for FAISS indexing. Returns (384,) or (d_model,) float32."""
        if self._native_ready:
            ids = self._tokenizer.Encode(text)[:16]
            if not ids:
                return np.zeros(self.d_model, dtype=np.float32)
            embedded = self._embedding[np.array(ids)]
            mean_embed = np.mean(embedded, axis=0)
            norm = np.linalg.norm(mean_embed) + 1e-8
            return (mean_embed / norm).astype(np.float32)
        else:
            return self._fallback.embed_query(text)
