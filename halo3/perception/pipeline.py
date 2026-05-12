"""Perception pipeline — orchestrates web fetch → embed → token tensor.

Uses the organism's own trained embedding when available (native perception).
Falls back to sentence-transformers if no LM checkpoint exists.
"""
from __future__ import annotations
import logging
import jax.numpy as jnp
import numpy as np
from halo3.perception.web_fetch import web_search, results_to_texts

log = logging.getLogger(__name__)


class PerceptionPipeline:
    """Fetch web content, embed, produce (n_tokens, d_model) tensor."""

    def __init__(self, d_model: int, n_tokens: int, vocab_size: int = 8000) -> None:
        self.n_tokens = n_tokens
        self.d_model = d_model

        # Try native embedder first (organism's own trained body)
        try:
            from halo3.perception.native_embedder import NativeEmbedder
            self.embedder = NativeEmbedder(d_model, vocab_size, n_tokens)
            if self.embedder._native_ready:
                log.info("Perception: using organism's OWN trained embedding")
            else:
                log.info("Perception: using sentence-transformers (no LM checkpoint yet)")
        except Exception:
            from halo3.perception.embedder import TextEmbedder
            self.embedder = TextEmbedder(d_model)
            log.info("Perception: using sentence-transformers fallback")

    def perceive(self, query: str, max_results: int = 5) -> tuple[jnp.ndarray, list[str]]:
        """Fetch and embed web content for a query."""
        results = web_search(query, max_results=max_results)
        if not results:
            log.warning(f"No results for '{query}', using zero tokens")
            return jnp.zeros((self.n_tokens, self.d_model)), []

        texts = results_to_texts(results)
        tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
        return tokens, texts

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for FAISS indexing."""
        return self.embedder.embed_query(query)
