"""Text embedder — sentence-transformers → d_model projection."""
from __future__ import annotations
import logging
import numpy as np
import jax.numpy as jnp

log = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384


class TextEmbedder:
    """Lazy-loads sentence-transformers, projects to d_model."""

    def __init__(self, d_model: int, seed: int = 42) -> None:
        self.d_model = d_model
        self._model = None
        # Fixed random projection (384 → d_model)
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((_EMBED_DIM, d_model)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-8

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(_MODEL_NAME)
                log.info(f"Loaded {_MODEL_NAME}")
            except ImportError:
                log.warning("sentence-transformers not installed, using random embeddings")
                self._model = "fallback"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed list of texts → (len(texts), d_model) float32."""
        self._load()
        if self._model == "fallback":
            # Deterministic fallback for testing
            rng = np.random.default_rng(hash(str(texts)) % 2**31)
            raw = rng.standard_normal((len(texts), _EMBED_DIM)).astype(np.float32)
        else:
            raw = self._model.encode(texts, convert_to_numpy=True)  # (N, 384)
        projected = raw @ self._proj  # (N, d_model)
        return projected.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed single text → (384,) for FAISS indexing."""
        self._load()
        if self._model == "fallback":
            rng = np.random.default_rng(hash(text) % 2**31)
            return rng.standard_normal(_EMBED_DIM).astype(np.float32)
        return self._model.encode([text], convert_to_numpy=True)[0]

    def texts_to_tokens(self, texts: list[str], n_tokens: int) -> jnp.ndarray:
        """Embed texts and pad/truncate to (n_tokens, d_model)."""
        if not texts:
            return jnp.zeros((n_tokens, self.d_model))
        embedded = self.embed_texts(texts)  # (N, d_model)
        if embedded.shape[0] >= n_tokens:
            return jnp.array(embedded[:n_tokens])
        # Pad with zeros
        pad = np.zeros((n_tokens - embedded.shape[0], self.d_model), dtype=np.float32)
        return jnp.array(np.concatenate([embedded, pad], axis=0))
