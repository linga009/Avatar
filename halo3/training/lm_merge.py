"""Load LM-trained weights into a fresh organism model."""
from __future__ import annotations
import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx

from halo3.config import Halo3Config
from halo3.backbone import Halo3Backbone
from halo3.lorentz_embedding import LorentzEmbedding

log = logging.getLogger(__name__)


def load_lm_into_model(model, cfg: Halo3Config, prefix: str = "data/checkpoints/halo3_lm"):
    """Replace backbone + lorentz_embed with LM-trained weights.

    The Hamiltonian, bridges, Kuramoto, and page memory remain freshly
    initialized — they will learn via per-tick prediction error.
    """
    bb_path = f"{prefix}_bb.eqx"
    le_path = f"{prefix}_le.eqx"

    if not os.path.exists(bb_path):
        raise FileNotFoundError(f"LM backbone not found: {bb_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"LM Lorentz embedding not found: {le_path}")

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    bb_template = Halo3Backbone(cfg, k1)
    le_template = LorentzEmbedding(cfg, k2)

    lm_bb = eqx.tree_deserialise_leaves(bb_path, bb_template)
    lm_le = eqx.tree_deserialise_leaves(le_path, le_template)

    model = eqx.tree_at(lambda m: m.backbone, model, lm_bb)
    model = eqx.tree_at(lambda m: m.lorentz_embed, model, lm_le)

    log.info(f"Loaded LM backbone from {bb_path}")
    log.info(f"Loaded LM Lorentz embedding from {le_path}")
    return model
