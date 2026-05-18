"""Unit tests for ParquetSource."""
from __future__ import annotations
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halo3.perception.web_fetch import SearchResult


def _make_parquet(dir_path: str) -> None:
    """Write a minimal test Parquet shard."""
    table = pa.table({
        "text": [
            "quantum computing uses quantum mechanics to process information efficiently",
            "photosynthesis converts sunlight into chemical energy in green plants",
            "neural networks learn complex patterns from large amounts of training data",
            "mitochondria is the powerhouse of the cell biology fundamentals",
            "gravitational waves detected by LIGO observatory Einstein relativity",
        ],
        "url": [
            "http://a.com", "http://b.com", "http://c.com",
            "http://d.com", "http://e.com",
        ],
        "int_score": [4, 3, 5, 4, 3],
    })
    pq.write_table(table, os.path.join(dir_path, "test.parquet"))


def test_loads_rows_and_builds_index():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        assert len(src._texts) == 5
        assert "quantum" in src._index


def test_search_returns_keyword_match():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        results = src.search("quantum computing", n=3)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert "quantum" in results[0].snippet.lower()


def test_search_sequential_fallback():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        results = src.search("zzzzzzzzz", n=2)  # no keyword match
        assert len(results) == 2
        assert src._cursor == 2


def test_filters_low_int_score():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        table = pa.table({
            "text": [
                "high quality educational content about science and learning",
                "low quality content here",
            ],
            "url": ["http://good.com", "http://bad.com"],
            "int_score": [4, 1],
        })
        pq.write_table(table, os.path.join(d, "test.parquet"))
        src = ParquetSource(d)
        assert len(src._texts) == 1
        assert "high quality" in src._texts[0]


def test_sample_texts():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        texts = src.sample_texts(3)
        assert len(texts) == 3
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)


def test_raises_if_no_parquet_files():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            ParquetSource(d)
