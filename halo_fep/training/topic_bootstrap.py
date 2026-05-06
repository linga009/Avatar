# halo_fep/training/topic_bootstrap.py
"""Wikipedia topic bootstrap — replaces synthetic MultimodalWorld data.

Streams WikiText-103 (a 103M-word Wikipedia corpus) via the HuggingFace
datasets library. Articles are filtered by per-cluster topic keywords and
embedded using the sentence-transformer CPU embedder, producing
(n_tokens, d_model) token arrays that match the heartbeat pipeline format.

Requires: pip install datasets sentence-transformers

The 8 topic clusters map to the HaloFEPConfig.n_hidden=8 hidden states,
establishing a clean semantic prior before the organism encounters noisy
live web data.
"""
from __future__ import annotations

import logging
import random
from itertools import cycle
from typing import Generator

import numpy as np

from halo_fep.config import HaloFEPConfig

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# Keywords that determine which cluster a Wikipedia article is routed to.
# These must cover all cfg.n_hidden=8 clusters (indices 0-7).
TOPIC_KEYWORDS: dict[int, list[str]] = {
    0: ["research", "study", "investigation", "findings", "experiment"],
    1: ["algorithm", "programming", "software", "api", "system", "network"],
    2: ["equation", "theorem", "proof", "mathematical", "calculus", "algebra"],
    3: ["philosophy", "theory", "ethics", "consciousness", "epistemology"],
    4: ["implementation", "code", "program", "function", "class", "compiler"],
    5: ["error", "failure", "problem", "diagnosis", "defect", "crash"],
    6: ["history", "historical", "century", "ancient", "civilization", "war"],
    7: ["future", "prediction", "forecast", "trend", "emerging", "innovation"],
}


def _text_to_tokens(
    text: str,
    n_tokens: int,
    d_model: int,
) -> np.ndarray:
    """Embed text into a (n_tokens, d_model) float32 token array.

    Splits the text into up to n_tokens equal-length chunks, embeds each
    chunk with a fixed random projection (no model load required for tests),
    and zero-pads remaining slots.

    In production this is called within iter_wikipedia_token_batches which
    uses the real sentence-transformer embedder.

    Args:
        text: Raw article text.
        n_tokens: Number of token slots (must match cfg.n_tokens).
        d_model: Embedding dimension (must match cfg.d_model).

    Returns:
        (n_tokens, d_model) float32 array.
    """
    tokens = np.zeros((n_tokens, d_model), dtype=np.float32)
    if not text.strip():
        return tokens

    # Split text into n_tokens chunks of equal character length
    chunk_size = max(1, len(text) // n_tokens)
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    for i, chunk in enumerate(chunks[:n_tokens]):
        if not chunk.strip():
            continue
        # Deterministic hash-based embedding (fallback used in tests)
        seed = abs(hash(chunk)) % (2 ** 31)
        rng  = np.random.RandomState(seed)
        vec  = rng.randn(d_model).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            tokens[i] = vec / norm

    return tokens


def _embed_chunk_real(text: str, model, d_model: int) -> np.ndarray:
    """Embed a single text chunk using a loaded SentenceTransformer."""
    raw = model.encode(text, normalize_embeddings=False, show_progress_bar=False)
    # raw is (384,); project to d_model via truncation or padding
    if len(raw) >= d_model:
        vec = raw[:d_model].astype(np.float32)
    else:
        vec = np.pad(raw, (0, d_model - len(raw))).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 1e-8 else vec


def iter_wikipedia_token_batches(
    cfg: HaloFEPConfig,
    seed: int = 42,
    articles_per_cluster: int = 200,
) -> Generator[np.ndarray, None, None]:
    """Yield (n_tokens, d_model) token arrays from WikiText-103 forever.

    Articles are filtered by TOPIC_KEYWORDS so each of the 8 clusters
    is represented roughly equally. The generator cycles indefinitely
    so it can drive an arbitrarily long bootstrap loop.

    Args:
        cfg: Config (n_tokens, d_model, n_hidden must be 8).
        seed: RNG seed for shuffling.
        articles_per_cluster: How many matching articles to buffer per cluster
                              before shuffling and cycling.

    Yields:
        (cfg.n_tokens, cfg.d_model) float32 numpy arrays.

    Raises:
        ImportError: If `datasets` is not installed.
        RuntimeError: If no articles match keywords for any cluster.
    """
    if load_dataset is None:
        raise ImportError(
            "Wikipedia topic bootstrap requires: pip install datasets"
        )

    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        use_real_embedder = True
        log.info("Using sentence-transformer for Wikipedia embeddings.")
    except Exception:
        st_model = None
        use_real_embedder = False
        log.warning(
            "sentence-transformers not available — using hash-based embeddings."
        )

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-v1",
        split="train",
        streaming=True,
    )

    # Collect articles_per_cluster matched articles per cluster
    buffers: dict[int, list[np.ndarray]] = {i: [] for i in range(cfg.n_hidden)}
    needed = {i: articles_per_cluster for i in range(cfg.n_hidden)}

    for article in dataset:
        text = article.get("text", "")
        if len(text) < 80:
            continue
        text_lower = text.lower()

        for cluster_idx, keywords in TOPIC_KEYWORDS.items():
            if needed[cluster_idx] <= 0:
                continue
            if any(kw in text_lower for kw in keywords):
                if use_real_embedder and st_model is not None:
                    chunk_size = max(1, len(text) // cfg.n_tokens)
                    chunks = [
                        text[i: i + chunk_size]
                        for i in range(0, len(text), chunk_size)
                    ]
                    tok = np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32)
                    for j, ch in enumerate(chunks[: cfg.n_tokens]):
                        tok[j] = _embed_chunk_real(ch, st_model, cfg.d_model)
                else:
                    tok = _text_to_tokens(text, cfg.n_tokens, cfg.d_model)

                buffers[cluster_idx].append(tok)
                needed[cluster_idx] -= 1
                break  # assign each article to at most one cluster

        if all(n <= 0 for n in needed.values()):
            break

    # Warn if any cluster is empty
    for ci, buf in buffers.items():
        if not buf:
            log.warning(
                f"No WikiText-103 articles matched cluster {ci} "
                f"({TOPIC_KEYWORDS[ci]}). Using zero tokens."
            )
            buffers[ci] = [np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32)]

    # Flatten, shuffle, and cycle indefinitely
    rng = random.Random(seed)
    all_tokens = [tok for buf in buffers.values() for tok in buf]
    rng.shuffle(all_tokens)
    log.info(f"Wikipedia bootstrap: {len(all_tokens)} articles buffered.")

    for tok in cycle(all_tokens):
        yield tok
