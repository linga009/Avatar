# halo_fep/training/hyperbolic_pretrain.py
"""WN18RR hyperbolic pre-training for the HoloEmbedding layer.

Pre-trains the HaloFEPModel's HoloEmbedding (Poincaré disk projection) on
WordNet IS-A hierarchy triples from WN18RR, using the Poincaré embedding
loss. Hierarchical relationships in WordNet map naturally to the Poincaré
disk's exponential distance growth near the boundary.

After pre-training, the HoloEmbedding produces geometrically structured
embeddings where hypernyms (general concepts) cluster near the disk center
and hyponyms (specific concepts) cluster near the boundary — improving the
HALO backbone's ability to represent hierarchical web knowledge.

Requires: pip install datasets

Usage:
    from halo_fep.training.hyperbolic_pretrain import run_hyperbolic_pretrain
    model = run_hyperbolic_pretrain(model, cfg, key)
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel

log = logging.getLogger(__name__)

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

_EPS = 1e-5


def poincare_distance(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Poincaré disk distance between two points u, v ∈ B^n (‖x‖ < 1).

    d(u, v) = arccosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))

    Both u and v are automatically clamped to the interior of the unit disk
    to prevent numerical instability near the boundary.

    Args:
        u: (..., dim) float32 point in the Poincaré disk.
        v: (..., dim) float32 point in the Poincaré disk.

    Returns:
        Scalar geodesic distance (non-negative float32).
    """
    # Clamp to strict interior: ‖x‖ < 1 - eps
    u = u / jnp.maximum(jnp.linalg.norm(u) + _EPS, 1.0 + _EPS)
    v = v / jnp.maximum(jnp.linalg.norm(v) + _EPS, 1.0 + _EPS)

    norm_u_sq = jnp.sum(u ** 2)
    norm_v_sq = jnp.sum(v ** 2)
    diff_sq   = jnp.sum((u - v) ** 2)

    alpha = 1.0 - norm_u_sq
    beta  = 1.0 - norm_v_sq

    arg = 1.0 + 2.0 * diff_sq / (alpha * beta + _EPS)
    # Clip to [1, inf) for valid arccosh domain; use 1.0 (not 1+eps) so that
    # identical points (diff_sq=0) produce distance 0 exactly.
    return jnp.arccosh(jnp.clip(arg, 1.0, None))


def poincare_loss(
    embeddings: jnp.ndarray,
    u_idx: int,
    v_idx: int,
    neg_idxs: jnp.ndarray,
) -> jnp.ndarray:
    """Poincaré embedding max-margin loss for one positive triple (u, v).

    Loss = -log[ exp(-d(u,v)) / (exp(-d(u,v)) + Σ_{v'} exp(-d(u,v'))) ]

    Minimizing this pulls linked entities (IS-A pairs) together on the disk
    while pushing negative samples apart.

    Args:
        embeddings: (n_entities, dim) float32 embedding table.
        u_idx: Head entity index.
        v_idx: Positive tail entity index (hypernym/hyponym).
        neg_idxs: (n_neg,) int32 negative tail indices.

    Returns:
        Scalar loss (non-negative float32).
    """
    u     = embeddings[u_idx]
    v_pos = embeddings[v_idx]

    d_pos = poincare_distance(u, v_pos)

    def d_neg_i(idx):
        return poincare_distance(u, embeddings[idx])

    d_neg = jax.vmap(d_neg_i)(neg_idxs)

    # Numerically stable log-softmax
    all_d  = jnp.concatenate([d_pos[None], d_neg])
    log_sm = jax.nn.log_softmax(-all_d)
    return -log_sm[0]   # maximize log-prob of positive pair


def run_hyperbolic_pretrain(
    model: HaloFEPModel,
    cfg: HaloFEPConfig,
    key: jnp.ndarray,
    n_steps: int = 1_000,
    n_entities: int = 40_943,   # WN18RR entity count
    n_negatives: int = 10,
    lr: float = 5e-3,
) -> HaloFEPModel:
    """Pre-train HoloEmbedding on WN18RR WordNet IS-A triples.

    Loads WN18RR via HuggingFace datasets (streaming=False, ~1MB download).
    Trains a (n_entities, d_boundary) Poincaré embedding table for n_steps
    gradient steps, then projects the learned embeddings onto the HoloEmbedding
    x_proj weight matrix via SVD-based alignment.

    Only the holo_embed.x_proj layer is modified; all other model components
    are unchanged.

    HoloEmbedding structure:
        x_proj: eqx.nn.Linear(d_model, d_boundary)  — weight shape (d_boundary, d_model)
        z_proj: eqx.nn.Linear(d_model, 1)

    The SVD alignment maps the principal directions of the learned Poincaré
    embedding table onto the rows of x_proj.weight, initialising the boundary
    projection with a geometry-aware basis.

    Args:
        model: HaloFEPModel to update.
        cfg: Config (d_boundary is the Poincaré embedding dimension).
        key: JAX PRNG key.
        n_steps: Number of gradient steps on WN18RR triples.
        n_entities: Number of WN18RR entities (40,943 in the full dataset).
        n_negatives: Number of negative samples per positive triple.
        lr: Learning rate for Poincaré SGD.

    Returns:
        Updated HaloFEPModel with improved HoloEmbedding weights.

    Raises:
        ImportError: If `datasets` is not installed.
    """
    if load_dataset is None:
        raise ImportError("WN18RR pre-training requires: pip install datasets")

    log.info("Loading WN18RR dataset...")
    triples = list(load_dataset("KGDatasets/WN18RR", split="train"))
    log.info(f"WN18RR: {len(triples)} training triples loaded.")

    # Initialize Poincaré embedding table in the interior of the disk
    key, ek = jax.random.split(key)
    d_emb = getattr(cfg, "d_boundary", cfg.d_model)
    raw = jax.random.normal(ek, (n_entities, d_emb)) * 0.01
    embeddings = raw  # start near origin (center of disk)

    opt = optax.sgd(lr)
    opt_state = opt.init(embeddings)

    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**30)))

    for step in range(n_steps):
        triple = triples[step % len(triples)]
        u_idx  = int(triple.get("head", triple.get("head_id", 0)) or 0)
        v_idx  = int(triple.get("tail", triple.get("tail_id", 1)) or 1)

        # Sample random negatives (corrupt the tail)
        neg_idxs = rng.integers(0, n_entities, size=n_negatives)
        neg_idxs_jnp = jnp.array(neg_idxs, dtype=jnp.int32)

        def loss_fn(emb):
            return poincare_loss(emb, u_idx, v_idx, neg_idxs_jnp)

        loss, grads = jax.value_and_grad(loss_fn)(embeddings)
        updates, opt_state_new = opt.update(grads, opt_state)
        opt_state = opt_state_new

        # Apply update then retract back into disk
        embeddings = embeddings + updates
        norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        scale = jnp.where(norms >= 1.0, (1.0 - _EPS) / norms, 1.0)
        embeddings = embeddings * scale

        if step % 200 == 0:
            log.info(f"WN18RR step {step}/{n_steps} | loss={float(loss):.4f}")

    # Align learned Poincaré embeddings with HoloEmbedding x_proj weight matrix via SVD.
    # x_proj.weight shape: (d_boundary, d_model) — maps d_model -> d_boundary.
    # We use the top d_boundary principal directions of the embedding table as the
    # new row basis for x_proj.weight, initialising it with WN18RR geometry.
    log.info("Aligning Poincaré embeddings with HoloEmbedding x_proj weight matrix...")
    emb_np = np.array(embeddings)                # (n_entities, d_emb)
    # Replace any NaN/Inf that can arise with very small entity tables (tests).
    emb_np = np.nan_to_num(emb_np, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(emb_np, full_matrices=False)
    # Vt shape: (min(n_entities, d_emb), d_emb) — right singular vectors

    d_model   = cfg.d_model
    # x_proj.weight has shape (d_boundary, d_model) = (d_emb, d_model)
    # Vt shape: (k, d_emb) where k = min(n_entities, d_emb).
    # We build a new weight matrix of shape (d_emb, d_model) by placing the
    # right singular vectors (rows of Vt, length d_emb) as columns, transposed.
    # Vt.T shape: (d_emb, k) — each column is a principal direction in d_emb space.
    Vt_T = Vt.T                    # (d_emb, k)
    k = Vt_T.shape[1]              # min(n_entities, d_emb)
    new_weight = np.zeros((d_emb, d_model), dtype=np.float32)
    cols_to_fill = min(k, d_model)
    new_weight[:, :cols_to_fill] = Vt_T[:, :cols_to_fill]

    # Update model.holo_embed.x_proj using eqx.tree_at
    try:
        model = eqx.tree_at(
            lambda m: m.holo_embed.x_proj.weight,
            model,
            jnp.array(new_weight),
        )
        log.info("HoloEmbedding x_proj weight updated with WN18RR Poincaré basis.")
    except Exception as e:
        log.warning(
            f"Could not update holo_embed.x_proj.weight ({e}). "
            "HoloEmbedding structure may differ — skipping alignment."
        )

    return model
