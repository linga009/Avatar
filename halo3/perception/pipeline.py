"""Perception pipeline — orchestrates web fetch → embed → token tensor."""
from __future__ import annotations
import logging
import jax.numpy as jnp
import numpy as np
from halo3.perception.embedder import TextEmbedder
from halo3.perception.web_fetch import web_search, results_to_texts

log = logging.getLogger(__name__)


class PerceptionPipeline:
    """Fetch web content, embed, produce (n_tokens, d_model) tensor."""

    def __init__(self, d_model: int, n_tokens: int) -> None:
        self.embedder = TextEmbedder(d_model)
        self.n_tokens = n_tokens
        self.d_model = d_model

    def perceive(self, query: str, max_results: int = 5) -> tuple[jnp.ndarray, list[str]]:
        """Fetch and embed web content for a query.

        Returns:
            tokens: (n_tokens, d_model) tensor for halo3_step
            texts: raw text chunks (for logging/debugging)
        """
        results = web_search(query, max_results=max_results)
        if not results:
            log.warning(f"No results for '{query}', using zero tokens")
            return jnp.zeros((self.n_tokens, self.d_model)), []

        texts = results_to_texts(results)
        tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
        return tokens, texts

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for FAISS indexing. Returns (384,) float32."""
        return self.embedder.embed_query(query)
