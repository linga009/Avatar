# halo_fep/intellect/goal_updater.py
"""Translates LLM GOAL: output into an update of model.gm.log_C.

Steps:
  1. Embed goal text → 384-dim (sentence-transformers, CPU)
  2. Project 384 → n_obs via random fixed projection (same approach as Embedder)
  3. Softmax → probability dist over preferred observations
  4. log → new log_C
  5. Replace via eqx.tree_at

Decay: each tick, C decays 1% toward uniform. Call decay(model) every step.
"""
from __future__ import annotations

import logging
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel

log = logging.getLogger(__name__)

_TEXT_DIM = 384


class GoalUpdater:
    def __init__(self, cfg: HaloFEPConfig, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((_TEXT_DIM, cfg.n_obs)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-8
        self._embedder = None  # lazy-init; tests may inject a mock

    def _get_embedder(self):
        if self._embedder is None:
            from halo_fep.perception.embedder import Embedder
            self._embedder = Embedder(d_model=self.cfg.d_model, seed=self.cfg.seed)
        return self._embedder

    def update_goal(self, model: HaloFEPModel, goal_text: str) -> HaloFEPModel:
        """Embed goal_text and update model.gm.log_C. Returns new model."""
        embedder  = self._get_embedder()
        text_emb  = embedder.embed_text(goal_text)                # (384,) float32
        logits    = text_emb @ self._proj                         # (n_obs,)
        probs     = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
        new_log_c = jnp.log(jnp.array(probs) + 1e-8)             # (n_obs,)
        return eqx.tree_at(lambda m: m.gm.log_C, model, new_log_c)

    def decay(self, model: HaloFEPModel, alpha: float = 0.99) -> HaloFEPModel:
        """Decay C matrix 1% toward uniform each step.

        Prevents the system from fixating on a single goal forever.
        """
        n_obs     = self.cfg.n_obs
        uniform   = jnp.full((n_obs,), -jnp.log(n_obs))  # log-uniform
        new_log_c = alpha * model.gm.log_C + (1.0 - alpha) * uniform
        return eqx.tree_at(lambda m: m.gm.log_C, model, new_log_c)
