# halo_fep/tests/test_episode_store.py
"""Tests for EpisodeStore — verifies Bug 5 fixes (batched FAISS writes, embed fallback).

Critical invariants tested
--------------------------
1. FAISS index is NOT written to disk on every insert (batched).
2. flush() forces a FAISS write to disk.
3. Semantic retrieval returns episodes ordered by similarity.
4. rebuild_index() reconstructs the FAISS index from SQLite correctly.
5. Zero-vector warning is emitted when no embed is supplied.
6. get_high_confidence() respects the min_delta threshold.
7. get_prioritized() returns IS weights in [0, 1].
"""
from __future__ import annotations

import os
import tempfile
import logging

import numpy as np
import pytest

from halo_fep.config import HaloFEPConfig
from halo_fep.memory.episode_store import EpisodeStore, _WRITE_EVERY
from halo_fep.memory.schema import Episode


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _make_episode(query: str = "test query", fe_delta: float = -0.1) -> Episode:
    cfg = HaloFEPConfig()
    return Episode(
        query             = query,
        tokens            = np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32),
        swarm_mu          = np.zeros((cfg.n_agents, cfg.n_hidden), dtype=np.float32),
        free_energy       = 1.0,
        free_energy_delta = fe_delta,
    )


def _random_embed(dim: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v   = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_faiss_not_written_on_every_insert(tmp_path):
    """FAISS index file should NOT be written after the first insert (< _WRITE_EVERY)."""
    store = EpisodeStore(str(tmp_path))
    idx_path = tmp_path / "faiss.index"

    ep    = _make_episode()
    embed = _random_embed()
    store.add(ep, query_embed=embed)

    # After 1 insert (_WRITE_EVERY = 50 by default), file should NOT yet exist
    # unless _WRITE_EVERY == 1.
    if _WRITE_EVERY > 1:
        assert not idx_path.exists(), (
            f"FAISS index was written after 1 insert (WRITE_EVERY={_WRITE_EVERY}). "
            "Batching is not working."
        )


def test_flush_writes_faiss_to_disk(tmp_path):
    """flush() must write the FAISS index to disk regardless of insert count."""
    store    = EpisodeStore(str(tmp_path))
    idx_path = tmp_path / "faiss.index"

    ep    = _make_episode()
    embed = _random_embed()
    store.add(ep, query_embed=embed)
    store.flush()

    assert idx_path.exists(), "flush() did not write FAISS index to disk"
    assert idx_path.stat().st_size > 0, "FAISS index file is empty after flush()"


def test_retrieve_returns_similar_episodes(tmp_path):
    """retrieve() should return the most similar episode first."""
    store = EpisodeStore(str(tmp_path))

    # Two distinct embeddings
    embed_a = _random_embed(seed=0)
    embed_b = _random_embed(seed=99)   # very different direction

    ep_a = _make_episode("topic A")
    ep_b = _make_episode("topic B")
    store.add(ep_a, query_embed=embed_a)
    store.add(ep_b, query_embed=embed_b)
    store.flush()

    results = store.retrieve(embed_a, k=2)
    assert len(results) == 2
    # First result should be ep_a (same embedding direction)
    assert results[0].query == "topic A", (
        f"Expected 'topic A' first but got '{results[0].query}'"
    )


def test_rebuild_index_round_trip(tmp_path):
    """rebuild_index() should reconstruct an index with the same ntotal as SQLite."""
    store = EpisodeStore(str(tmp_path))
    n = 5
    for i in range(n):
        store.add(_make_episode(f"q{i}"), query_embed=_random_embed(seed=i))
    store.flush()

    original_total = store._index.ntotal
    store.rebuild_index()

    assert store._index.ntotal == original_total, (
        f"rebuild_index produced {store._index.ntotal} vectors, "
        f"expected {original_total}"
    )
    assert len(store._ids) == original_total


def test_no_embed_emits_warning(tmp_path, caplog):
    """Calling add() without query_embed should emit a WARNING."""
    store = EpisodeStore(str(tmp_path))
    ep    = _make_episode()
    with caplog.at_level(logging.WARNING, logger="halo_fep.memory.episode_store"):
        store.add(ep, query_embed=None)
    assert any("query_embed" in r.message for r in caplog.records), (
        "No warning emitted when query_embed=None"
    )


def test_get_high_confidence_threshold(tmp_path):
    """Only episodes with free_energy_delta < min_delta should be returned."""
    store = EpisodeStore(str(tmp_path))
    store.add(_make_episode(fe_delta=-0.2), query_embed=_random_embed(seed=0))
    store.add(_make_episode(fe_delta=-0.01), query_embed=_random_embed(seed=1))  # should be excluded
    store.add(_make_episode(fe_delta=-0.5),  query_embed=_random_embed(seed=2))

    results = store.get_high_confidence(min_delta=-0.05)
    assert len(results) == 2
    for ep in results:
        assert ep.free_energy_delta < -0.05


def test_get_prioritized_weights_in_range(tmp_path):
    """IS weights from get_prioritized() must all be in [0, 1]."""
    store = EpisodeStore(str(tmp_path))
    for i in range(10):
        store.add(_make_episode(fe_delta=-(i + 1) * 0.05), query_embed=_random_embed(seed=i))

    episodes, weights = store.get_prioritized(n=5)
    assert len(episodes) == len(weights)
    assert np.all(weights >= 0.0) and np.all(weights <= 1.0 + 1e-6), (
        f"IS weights out of [0, 1] range: min={weights.min():.4f}, max={weights.max():.4f}"
    )


def test_update_llm_output(tmp_path):
    """update_llm_output should update the stored episode's llm_output field."""
    store = EpisodeStore(str(tmp_path))
    ep    = _make_episode()
    store.add(ep, query_embed=_random_embed())
    store.flush()

    store.update_llm_output(ep.id, "SEARCH: new query")
    results = store.get_recent(n=1)
    assert results[0].llm_output == "SEARCH: new query"
