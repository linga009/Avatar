"""Tests for TopicIndex — topic clustering and streaming retrieval."""
from __future__ import annotations
import json
import os
import tempfile
import pytest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


# ---------- build() tests ----------


def _make_parquet(path: str, n_rows: int = 100) -> str:
    """Create a minimal parquet file with text and int_score columns."""
    topics = ["quantum physics laser photon experiment",
              "biology cell membrane protein enzyme",
              "history ancient rome emperor civilization",
              "machine learning neural network gradient descent"]
    texts = []
    scores = []
    for i in range(n_rows):
        base = topics[i % len(topics)]
        texts.append(f"{base} row {i} with some extra words for length")
        scores.append(4 if i % 5 != 0 else 2)  # 80% pass score filter

    table = pa.table({
        "text": texts,
        "int_score": scores,
        "url": [f"http://example.com/{i}" for i in range(n_rows)],
    })
    parquet_path = os.path.join(path, "sample", "10BT", "test_shard.parquet")
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    pq.write_table(table, parquet_path, row_group_size=25)
    return path


def test_build_index():
    """TopicIndex.build creates a valid index from parquet files."""
    from halo3.perception.topic_index import TopicIndex

    with tempfile.TemporaryDirectory() as td:
        parquet_dir = _make_parquet(td, n_rows=200)
        index_path = os.path.join(td, "topic_index.json")
        idx = TopicIndex.build(parquet_dir, index_path)

        topics = idx.get_topics()
        assert len(topics) >= 3  # at least 3 distinct topic clusters
        total_positions = sum(t.row_count for t in topics)
        assert total_positions >= 100  # most rows assigned (80% pass score filter)

        # Verify index file was written
        assert os.path.exists(index_path)

        # Verify we can reload from the saved file
        idx2 = TopicIndex(index_path, parquet_dir)
        assert len(idx2.get_topics()) == len(topics)


def test_build_index_streaming():
    """sample_from_topic returns actual text from parquet after build."""
    from halo3.perception.topic_index import TopicIndex

    with tempfile.TemporaryDirectory() as td:
        parquet_dir = _make_parquet(td, n_rows=100)
        index_path = os.path.join(td, "topic_index.json")
        idx = TopicIndex.build(parquet_dir, index_path)

        topics = idx.get_topics()
        assert len(topics) >= 1
        texts = idx.sample_from_topic(topics[0].topic_id, n=3)
        assert len(texts) >= 1
        assert all(isinstance(t, str) and len(t) > 10 for t in texts)
