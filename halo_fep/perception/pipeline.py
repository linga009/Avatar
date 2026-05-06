"""PerceptionPipeline: web search → (n_tokens, d_model) JAX array."""
from __future__ import annotations

import logging
import numpy as np
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.perception.web_fetcher import WebFetcher
from halo_fep.perception.embedder import Embedder
from halo_fep.perception.token_packer import pack_results

log = logging.getLogger(__name__)


class PerceptionPipeline:
    def __init__(self, cfg: HaloFEPConfig) -> None:
        self.cfg      = cfg
        self.fetcher  = WebFetcher()
        self.embedder = Embedder(d_model=cfg.d_model, seed=cfg.seed)

    def embed(self, query: str) -> jnp.ndarray:
        """Search query, embed results, return (n_tokens, d_model) float32."""
        results     = self.fetcher.search(query, max_results=5)
        query_embed = self.embedder.embed_text(query)
        tokens_np   = pack_results(
            query_embed, results, self.embedder,
            n_tokens=self.cfg.n_tokens, d_model=self.cfg.d_model,
        )
        return jnp.array(tokens_np)

    def embed_query(self, query: str) -> np.ndarray:
        """Returns (d_model,) numpy float32 for FAISS retrieval."""
        return self.embedder.embed_text(query)

    def query_from_beliefs(self, carry) -> str:
        """Convert dominant belief cluster + action to a search query string.

        Reads argmax of mean swarm_mu across agents as the belief index,
        and argmax of mean swarm_action as the action index.
        Returns a templated query string like "topic 3 action 1 learning".
        """
        import jax.numpy as jnp
        mean_mu     = jnp.mean(carry.swarm_mu, axis=0)      # (n_hidden,)
        belief_idx  = int(jnp.argmax(mean_mu))
        mean_action = jnp.mean(carry.swarm_action, axis=0)  # (n_actions,)
        action_idx  = int(jnp.argmax(mean_action))
        templates = [
            f"topic {belief_idx} research exploration",
            f"concept {belief_idx} deep learning",
            f"idea {belief_idx} artificial intelligence",
            f"theory {belief_idx} neural network",
        ]
        return templates[action_idx % len(templates)]
