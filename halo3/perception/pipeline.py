"""Perception pipeline — orchestrates fetch -> embed -> token tensor.

v3.11: Uses TopicIndex for FE-guided active learning from local parquet data.
Falls back to web search when no local data or topic index is present.
"""
from __future__ import annotations
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
        self._topic_index = None

        # Auto-detect TopicIndex
        index_path = "data/fineweb/topic_index.json"
        if os.path.exists(index_path):
            try:
                from halo3.perception.topic_index import TopicIndex
                self._topic_index = TopicIndex(index_path, "data/fineweb")
                log.info("Perception: using TopicIndex (FE-guided active learning)")
            except Exception as e:
                log.warning(f"TopicIndex failed to load: {e} — falling back to web search")

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

        if self._topic_index is None:
            log.info("Perception: using web search (no TopicIndex found)")

    def perceive(
        self,
        query: str,
        max_results: int = 5,
        model=None,
        carry=None,
        key=None,
    ) -> tuple[jnp.ndarray, list[str]]:
        """Fetch and embed content for a query.

        When TopicIndex is available and model/carry are provided, uses
        FE scoring to pick the most informative texts. Otherwise falls
        back to keyword matching or web search.
        """
        if self._topic_index is not None:
            topics = self._topic_index.match_topic(query)
            if topics:
                # Stream candidates from matching topics
                n_candidates = max_results * 3
                candidates = self._topic_index.sample_from_topics(
                    [t.topic_id for t in topics[:5]],
                    n_per_topic=max(1, n_candidates // min(5, len(topics))),
                )

                if candidates:
                    # FE-rank if model available
                    if model is not None and carry is not None and key is not None:
                        candidates = self._fe_rank(candidates, model, carry, key, max_results)
                    else:
                        candidates = candidates[:max_results]

                    if candidates:
                        tokens = self.embedder.texts_to_tokens(candidates, self.n_tokens)
                        return tokens, candidates

            # TopicIndex found no matches — fall through to web search
            log.debug(f"TopicIndex: no matches for '{query[:30]}' — trying web search")

        # Fallback: web search
        from halo3.perception.web_fetch import web_search, results_to_texts
        results = web_search(query, max_results=max_results)

        if not results:
            log.warning(f"No results for '{query}', using zero tokens")
            return jnp.zeros((self.n_tokens, self.d_model)), []

        texts = results_to_texts(results)
        tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
        return tokens, texts

    def _fe_rank(
        self,
        candidates: list[str],
        model,
        carry,
        key,
        n_select: int,
    ) -> list[str]:
        """Rank candidates by forward-only free energy, return zone of proximal development."""
        from halo3.loss import halo3_loss
        from halo3.training.active_sampler import select_texts_by_fe

        fe_scores = []
        for text in candidates:
            tokens = self.embedder.texts_to_tokens([text], self.n_tokens)
            loss, _ = halo3_loss(model, carry, tokens, key)
            fe_scores.append(float(loss))

        return select_texts_by_fe(candidates, fe_scores, n_select=n_select)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for FAISS indexing."""
        return self.embedder.embed_query(query)
