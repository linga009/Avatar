"""Tests for TopicIndex — topic clustering and streaming retrieval."""
from __future__ import annotations
import json
import os
import tempfile
import pytest


def _make_index_json(path: str, n_topics: int = 3) -> str:
    """Create a minimal topic_index.json for testing."""
    topics = []
    for i in range(n_topics):
        topics.append({
            "topic_id": i,
            "keywords": [f"word{i}a", f"word{i}b", f"word{i}c"],
            "row_count": 10 + i,
            "sample_titles": [f"Title {i}-{j}" for j in range(3)],
            "file_positions": [
                {"file": 0, "row_group": 0, "row": i * 10 + j}
                for j in range(10 + i)
            ],
        })
    index_path = os.path.join(path, "topic_index.json")
    with open(index_path, "w") as f:
        json.dump({"topics": topics, "parquet_files": ["shard.parquet"]}, f)
    return index_path


def test_load_index():
    from halo3.perception.topic_index import TopicIndex
    with tempfile.TemporaryDirectory() as td:
        index_path = _make_index_json(td, n_topics=5)
        idx = TopicIndex(index_path, parquet_dir=td)
        topics = idx.get_topics()
        assert len(topics) == 5
        assert topics[0].topic_id == 0
        assert topics[0].keywords == ["word0a", "word0b", "word0c"]
        assert topics[0].row_count == 10


def test_match_topic():
    from halo3.perception.topic_index import TopicIndex
    with tempfile.TemporaryDirectory() as td:
        index_path = _make_index_json(td, n_topics=5)
        idx = TopicIndex(index_path, parquet_dir=td)
        matches = idx.match_topic("word2a word2b something")
        assert len(matches) >= 1
        assert matches[0].topic_id == 2


def test_match_topic_no_match_returns_empty():
    from halo3.perception.topic_index import TopicIndex
    with tempfile.TemporaryDirectory() as td:
        index_path = _make_index_json(td, n_topics=3)
        idx = TopicIndex(index_path, parquet_dir=td)
        matches = idx.match_topic("xyzzy plugh")
        assert matches == []
