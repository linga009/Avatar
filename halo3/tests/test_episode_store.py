"""Tests for EpisodeStore island summary methods (Task 3 — memory pipeline).

TDD: written before implementation. All four tests should fail until
save_island_summary / get_island_summaries / retrieve_similar_island are added.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from halo3.memory.episode_store import EpisodeStore


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_store(tmpdir: str) -> EpisodeStore:
    db_path = os.path.join(tmpdir, "test_episodes.db")
    return EpisodeStore(db_path=db_path)


def _unit_vec(dim: int, index: int) -> np.ndarray:
    """Return a float32 unit vector with a 1.0 at position `index`."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# test_save_island_summary
# ---------------------------------------------------------------------------

def test_save_island_summary():
    """Store a summary and retrieve it; verify array values and topic roundtrip."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = _make_store(tmpdir)
        try:
            summary = np.random.default_rng(42).random(2048).astype(np.float32)
            carry = np.random.default_rng(7).random(128).astype(np.float32)
            tick = 100
            topic = "quantum_cognition"

            store.save_island_summary(tick, summary, carry, topic)

            rows = store.get_island_summaries()
            assert len(rows) == 1

            row = rows[0]
            assert row["tick"] == tick
            assert row["topic"] == topic
            np.testing.assert_array_almost_equal(row["summary"], summary)
            np.testing.assert_array_almost_equal(row["carry_embedding"], carry)
        finally:
            store.conn.close()


# ---------------------------------------------------------------------------
# test_retrieve_similar_island
# ---------------------------------------------------------------------------

def test_retrieve_similar_island():
    """Two summaries with distinct carry embeddings; query near one returns that one."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = _make_store(tmpdir)
        try:
            carry_a = _unit_vec(128, 0)   # points along axis 0
            carry_b = _unit_vec(128, 1)   # points along axis 1
            summary = np.ones(2048, dtype=np.float32)

            store.save_island_summary(10, summary, carry_a, "topic_a")
            store.save_island_summary(20, summary, carry_b, "topic_b")

            # Query very close to carry_a
            query = _unit_vec(128, 0)
            result = store.retrieve_similar_island(query, threshold=0.3)

            assert result is not None
            assert result["topic"] == "topic_a"
            assert result["tick"] == 10
        finally:
            store.conn.close()


# ---------------------------------------------------------------------------
# test_retrieve_similar_island_below_threshold
# ---------------------------------------------------------------------------

def test_retrieve_similar_island_below_threshold():
    """Orthogonal query embedding returns None when similarity < threshold."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = _make_store(tmpdir)
        try:
            carry = _unit_vec(128, 0)   # points along axis 0
            summary = np.ones(2048, dtype=np.float32)
            store.save_island_summary(5, summary, carry, "some_topic")

            # Orthogonal to carry → cosine similarity = 0, below default threshold 0.3
            query = _unit_vec(128, 1)
            result = store.retrieve_similar_island(query, threshold=0.3)

            assert result is None
        finally:
            store.conn.close()


# ---------------------------------------------------------------------------
# test_retrieve_similar_island_empty
# ---------------------------------------------------------------------------

def test_retrieve_similar_island_empty():
    """No summaries stored → retrieve_similar_island returns None."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        store = _make_store(tmpdir)
        try:
            query = np.random.default_rng(0).random(128).astype(np.float32)
            result = store.retrieve_similar_island(query)

            assert result is None
        finally:
            store.conn.close()
