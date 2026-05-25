"""Integration test: TopicIndex build -> ActiveSampler -> text selection."""
from __future__ import annotations
import os
import tempfile
import pytest
import pyarrow as pa
import pyarrow.parquet as pq


def _make_corpus(path: str, n_rows: int = 200) -> str:
    """Create a parquet corpus with distinct topic clusters."""
    topic_texts = {
        "physics": "quantum mechanics entanglement photon coherence wave function particle",
        "biology": "cell membrane protein enzyme mitochondria genetics chromosome",
        "history": "ancient rome civilization emperor republic senate colosseum",
        "computer": "algorithm neural network gradient descent optimization machine learning",
    }
    texts, scores = [], []
    for i in range(n_rows):
        topic_name = list(topic_texts.keys())[i % 4]
        base = topic_texts[topic_name]
        texts.append(f"{base} document {i} additional content here for indexing")
        scores.append(4)

    table = pa.table({"text": texts, "int_score": scores, "url": [""] * n_rows})
    parquet_dir = os.path.join(path, "sample", "10BT")
    os.makedirs(parquet_dir, exist_ok=True)
    pq.write_table(table, os.path.join(parquet_dir, "corpus.parquet"), row_group_size=50)
    return path


def test_full_pipeline():
    """Build index, query topics, sample texts — end-to-end."""
    from halo3.perception.topic_index import TopicIndex

    with tempfile.TemporaryDirectory() as td:
        parquet_dir = _make_corpus(td, n_rows=200)
        index_path = os.path.join(td, "topic_index.json")

        # Build
        idx = TopicIndex.build(parquet_dir, index_path)
        topics = idx.get_topics()
        assert len(topics) >= 3

        # Query — physics should match
        matches = idx.match_topic("quantum photon entanglement")
        assert len(matches) >= 1
        top_kw = set(matches[0].keywords)
        assert "quantum" in top_kw or "photon" in top_kw or "entanglement" in top_kw

        # Stream texts from disk
        texts = idx.sample_from_topic(matches[0].topic_id, n=5)
        assert len(texts) >= 1
        assert any("quantum" in t or "photon" in t for t in texts)


def test_fe_selection_filters_correctly():
    """select_texts_by_fe keeps medium-FE texts, drops extremes."""
    from halo3.training.active_sampler import select_texts_by_fe

    texts = [f"doc_{i}" for i in range(50)]
    fe_scores = [float(i) for i in range(50)]

    selected = select_texts_by_fe(texts, fe_scores, n_select=10)
    assert len(selected) == 10

    selected_indices = {int(t.split("_")[1]) for t in selected}
    assert all(idx >= 10 for idx in selected_indices), "Low-FE texts should be filtered"
    assert all(idx < 40 for idx in selected_indices), "High-FE texts should be filtered"
