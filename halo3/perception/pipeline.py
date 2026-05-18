"""Perception pipeline — orchestrates fetch → embed → token tensor.

Uses FineWeb-Edu ParquetSource when data/fineweb/ contains Parquet files.
Falls back to web search when no local data is present.
"""
from __future__ import annotations
import glob
import logging
import os

import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


class PerceptionPipeline:
    """Fetch content, embed, produce (n_tokens, d_model) tensor."""

    def __init__(self, d_model: int, n_tokens: int, vocab_size: int = 8000) -> None:
        self.n_tokens = n_tokens
        self.d_model = d_model
        self._parquet = None

        # Auto-detect FineWeb-Edu Parquet source
        parquet_dir = "data/fineweb"
        if glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True):
            try:
                from halo3.perception.parquet_source import ParquetSource
                self._parquet = ParquetSource(parquet_dir)
                log.info("Perception: using FineWeb-Edu Parquet source (web search disabled)")
            except Exception as e:
                log.warning(f"ParquetSource failed to load: {e} — falling back to web search")

        # Embedder (always needed)
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

        if self._parquet is None:
            log.info("Perception: using web search (no FineWeb data found)")

    def perceive(self, query: str, max_results: int = 5) -> tuple[jnp.ndarray, list[str]]:
        """Fetch and embed content for a query."""
        if self._parquet is not None:
            results = self._parquet.search(query, n=max_results)
        else:
            from halo3.perception.web_fetch import web_search
            results = web_search(query, max_results=max_results)

        if not results:
            log.warning(f"No results for '{query}', using zero tokens")
            return jnp.zeros((self.n_tokens, self.d_model)), []

        from halo3.perception.web_fetch import results_to_texts
        texts = results_to_texts(results)
        tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
        return tokens, texts

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for FAISS indexing."""
        return self.embedder.embed_query(query)
