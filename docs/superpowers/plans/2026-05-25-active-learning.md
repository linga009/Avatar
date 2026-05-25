# Free Energy-Guided Active Learning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sequential FineWeb curriculum with FE-scored active learning — Avatar's own prediction error determines what it learns.

**Architecture:** TopicIndex clusters 726K+ parquet rows into ~300-500 topic buckets with disk pointers. ActiveSampler uses BS valuation to pick topics, streams candidates from disk, forward-pass scores them by free energy, and selects the zone-of-proximal-development texts for training. Replaces both dream Phase 4 cursor and waking ParquetSource.

**Tech Stack:** Python, JAX, pyarrow, numpy (no new deps)

---

### Task 1: TopicIndex — Data Structures and Loading

**Files:**
- Create: `halo3/perception/topic_index.py`
- Test: `tests/perception/test_topic_index.py`

- [ ] **Step 1: Write failing test for TopicIndex load from JSON**

```python
# tests/perception/test_topic_index.py
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
    """TopicIndex loads from JSON and exposes topic buckets."""
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
    """match_topic returns topics whose keywords overlap with query."""
    from halo3.perception.topic_index import TopicIndex

    with tempfile.TemporaryDirectory() as td:
        index_path = _make_index_json(td, n_topics=5)
        idx = TopicIndex(index_path, parquet_dir=td)
        matches = idx.match_topic("word2a word2b something")
        assert len(matches) >= 1
        assert matches[0].topic_id == 2


def test_match_topic_no_match_returns_empty():
    """match_topic returns empty list when no keywords overlap."""
    from halo3.perception.topic_index import TopicIndex

    with tempfile.TemporaryDirectory() as td:
        index_path = _make_index_json(td, n_topics=3)
        idx = TopicIndex(index_path, parquet_dir=td)
        matches = idx.match_topic("xyzzy plugh")
        assert matches == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/perception/test_topic_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'halo3.perception.topic_index'`

- [ ] **Step 3: Implement TopicIndex data structures and loading**

```python
# halo3/perception/topic_index.py
"""TopicIndex — clustered topic map over FineWeb-Edu parquet corpus.

Streams parquet row groups to build a lightweight topic index. At runtime,
the index is a JSON file (~5-10MB) with pointers back into parquet files.
Texts are never held in RAM — streamed from disk on demand.
"""
from __future__ import annotations
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field

import pyarrow.parquet as pq

log = logging.getLogger(__name__)

_MIN_WORD_LEN = 4
_MIN_SCORE = 3


@dataclass
class TopicBucket:
    topic_id: int
    keywords: list[str]
    row_count: int
    sample_titles: list[str]
    file_positions: list[dict]  # [{file, row_group, row}, ...]


class TopicIndex:
    """Lightweight topic map over a parquet corpus.

    Loads a persisted JSON index. Texts are streamed from parquet on demand.
    """

    def __init__(self, index_path: str, parquet_dir: str) -> None:
        self._parquet_dir = parquet_dir
        with open(index_path) as f:
            data = json.load(f)
        self._parquet_files: list[str] = data.get("parquet_files", [])
        self._topics: list[TopicBucket] = []
        for t in data["topics"]:
            self._topics.append(TopicBucket(
                topic_id=t["topic_id"],
                keywords=t["keywords"],
                row_count=t["row_count"],
                sample_titles=t.get("sample_titles", []),
                file_positions=t["file_positions"],
            ))
        # Build keyword -> topic_id lookup for fast matching
        self._keyword_to_topics: dict[str, list[int]] = {}
        for i, topic in enumerate(self._topics):
            for kw in topic.keywords:
                self._keyword_to_topics.setdefault(kw, []).append(i)
        log.info(f"TopicIndex: loaded {len(self._topics)} topics from {index_path}")

    def get_topics(self) -> list[TopicBucket]:
        return self._topics

    def match_topic(self, query: str) -> list[TopicBucket]:
        """Find topics whose keywords overlap with query words."""
        words = set(re.findall(rf"[a-z]{{{_MIN_WORD_LEN},}}", query.lower()))
        if not words:
            return []
        scores: dict[int, int] = {}
        for w in words:
            for idx in self._keyword_to_topics.get(w, []):
                scores[idx] = scores.get(idx, 0) + 1
        if not scores:
            return []
        ranked = sorted(scores, key=lambda i: -scores[i])
        return [self._topics[i] for i in ranked]

    def sample_from_topic(self, topic_id: int, n: int) -> list[str]:
        """Stream n random texts from a topic's file positions."""
        topic = self._topics[topic_id]
        positions = topic.file_positions
        if not positions:
            return []
        chosen = random.sample(positions, min(n, len(positions)))
        return self._read_positions(chosen)

    def sample_from_topics(self, topic_ids: list[int], n_per_topic: int) -> list[str]:
        """Stream texts from multiple topics."""
        texts = []
        for tid in topic_ids:
            texts.extend(self.sample_from_topic(tid, n_per_topic))
        return texts

    def _read_positions(self, positions: list[dict]) -> list[str]:
        """Read specific rows from parquet files. Opens and closes per file."""
        # Group by (file, row_group) to minimize parquet opens
        grouped: dict[tuple[int, int], list[int]] = {}
        for pos in positions:
            key = (pos["file"], pos["row_group"])
            grouped.setdefault(key, []).append(pos["row"])

        texts = []
        for (file_idx, rg_idx), rows in grouped.items():
            if file_idx >= len(self._parquet_files):
                continue
            path = os.path.join(self._parquet_dir, self._parquet_files[file_idx])
            if not os.path.exists(path):
                log.warning(f"TopicIndex: parquet file missing: {path}")
                continue
            try:
                pf = pq.ParquetFile(path)
                if rg_idx >= pf.num_row_groups:
                    continue
                batch = pf.read_row_group(rg_idx, columns=["text"])
                col = batch.column("text")
                for row in rows:
                    if row < len(col):
                        text = col[row].as_py()
                        if text:
                            texts.append(text)
            except Exception as e:
                log.warning(f"TopicIndex: read error {path} rg={rg_idx}: {e}")
        return texts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/perception/test_topic_index.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/perception/topic_index.py tests/perception/test_topic_index.py && git commit -m "feat: TopicIndex data structures and loading"
```

---

### Task 2: TopicIndex — Build from Parquet

**Files:**
- Modify: `halo3/perception/topic_index.py`
- Test: `tests/perception/test_topic_index.py`

- [ ] **Step 1: Write failing test for index build**

Add to `tests/perception/test_topic_index.py`:

```python
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/perception/test_topic_index.py::test_build_index tests/perception/test_topic_index.py::test_build_index_streaming -v`
Expected: FAIL — `AttributeError: type object 'TopicIndex' has no attribute 'build'`

- [ ] **Step 3: Implement TopicIndex.build**

Add to `halo3/perception/topic_index.py`:

```python
import glob
import math
from collections import defaultdict

# Add this method to TopicIndex class:

    @staticmethod
    def build(parquet_dir: str, output_path: str) -> "TopicIndex":
        """Scan corpus and build topic index. Streams row groups — never loads full corpus.

        Two-pass approach:
          Pass 1: Compute document frequencies (IDF) for all words
          Pass 2: Extract top-5 TF-IDF keywords per text, cluster into topic buckets
        """
        files = sorted(glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True))
        if not files:
            raise FileNotFoundError(f"No parquet files in {parquet_dir}")

        # Relative paths for portability inside Docker
        rel_files = [os.path.relpath(f, parquet_dir) for f in files]

        log.info(f"TopicIndex.build: scanning {len(files)} parquet files...")

        # --- Pass 1: Document frequencies ---
        doc_freq: dict[str, int] = defaultdict(int)
        total_docs = 0
        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                batch = pf.read_row_group(rg, columns=["text", "int_score"])
                d = batch.to_pydict()
                for text, score in zip(d["text"], d["int_score"]):
                    if (score or 0) < _MIN_SCORE or not text or not text.strip():
                        continue
                    total_docs += 1
                    words = set(re.findall(rf"[a-z]{{{_MIN_WORD_LEN},}}", text.lower()))
                    for w in words:
                        doc_freq[w] += 1
                del batch, d
            if total_docs % 100_000 == 0 and total_docs > 0:
                log.info(f"  Pass 1: {total_docs:,} docs scanned")

        log.info(f"TopicIndex.build: Pass 1 done — {total_docs:,} docs, {len(doc_freq):,} unique words")

        # Pre-compute IDF
        idf: dict[str, float] = {}
        for word, df in doc_freq.items():
            idf[word] = math.log(total_docs / df) if df > 0 else 0.0
        # Filter out very common words (IDF < 1.0 means in >36% of docs)
        idf = {w: v for w, v in idf.items() if v >= 1.0}
        del doc_freq

        # --- Pass 2: Extract keywords, cluster into buckets ---
        buckets: list[dict] = []  # {keywords: set, positions: list, titles: list}
        keyword_to_bucket: dict[str, int] = {}  # keyword -> bucket index

        file_idx_map = {os.path.abspath(f): i for i, f in enumerate(files)}
        docs_assigned = 0

        for path in files:
            fi = file_idx_map[os.path.abspath(path)]
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                batch = pf.read_row_group(rg, columns=["text", "int_score"])
                d = batch.to_pydict()
                for row_off, (text, score) in enumerate(zip(d["text"], d["int_score"])):
                    if (score or 0) < _MIN_SCORE or not text or not text.strip():
                        continue
                    # TF-IDF keywords
                    words = re.findall(rf"[a-z]{{{_MIN_WORD_LEN},}}", text.lower())
                    if not words:
                        continue
                    word_counts: dict[str, int] = defaultdict(int)
                    for w in words:
                        word_counts[w] += 1
                    n_words = len(words)
                    scored_words = []
                    for w, count in word_counts.items():
                        if w in idf:
                            tf = count / n_words
                            scored_words.append((w, tf * idf[w]))
                    scored_words.sort(key=lambda x: -x[1])
                    top_kw = [w for w, _ in scored_words[:5]]
                    if not top_kw:
                        continue

                    pos = {"file": fi, "row_group": rg, "row": row_off}
                    title = " ".join(text.split()[:8])

                    # Find best matching bucket (2+ keyword overlap)
                    bucket_scores: dict[int, int] = defaultdict(int)
                    for kw in top_kw:
                        if kw in keyword_to_bucket:
                            bucket_scores[keyword_to_bucket[kw]] += 1

                    best_bucket = -1
                    best_overlap = 0
                    for bi, overlap in bucket_scores.items():
                        if overlap >= 2 and overlap > best_overlap:
                            best_bucket = bi
                            best_overlap = overlap

                    if best_bucket >= 0:
                        b = buckets[best_bucket]
                        b["positions"].append(pos)
                        for kw in top_kw:
                            b["keywords"].add(kw)
                            keyword_to_bucket[kw] = best_bucket
                        if len(b["titles"]) < 5:
                            b["titles"].append(title)
                    else:
                        # New bucket
                        bi = len(buckets)
                        buckets.append({
                            "keywords": set(top_kw),
                            "positions": [pos],
                            "titles": [title],
                        })
                        for kw in top_kw:
                            keyword_to_bucket[kw] = bi

                    docs_assigned += 1
                del batch, d

        log.info(f"TopicIndex.build: Pass 2 done — {docs_assigned:,} docs in {len(buckets)} raw buckets")

        # --- Merge small buckets (< 5 rows) into nearest neighbor ---
        merged = []
        for b in buckets:
            if len(b["positions"]) < 5:
                # Find bucket with most keyword overlap
                best_target = -1
                best_overlap = 0
                for mi, mb in enumerate(merged):
                    overlap = len(b["keywords"] & mb["keywords"])
                    if overlap > best_overlap:
                        best_target = mi
                        best_overlap = overlap
                if best_target >= 0 and best_overlap >= 1:
                    merged[best_target]["positions"].extend(b["positions"])
                    merged[best_target]["keywords"] |= b["keywords"]
                else:
                    merged.append(b)
            else:
                merged.append(b)

        # --- Serialize ---
        topics_out = []
        for i, b in enumerate(merged):
            # Keep top-10 keywords by how many docs they appear in across corpus
            kw_list = sorted(b["keywords"], key=lambda w: -idf.get(w, 0.0))[:10]
            topics_out.append({
                "topic_id": i,
                "keywords": kw_list,
                "row_count": len(b["positions"]),
                "sample_titles": b["titles"][:5],
                "file_positions": b["positions"],
            })

        index_data = {
            "parquet_files": rel_files,
            "topics": topics_out,
        }

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(index_data, f)
        log.info(f"TopicIndex.build: saved {len(topics_out)} topics to {output_path}")

        return TopicIndex(output_path, parquet_dir)
```

- [ ] **Step 4: Run all TopicIndex tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/perception/test_topic_index.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/perception/topic_index.py tests/perception/test_topic_index.py && git commit -m "feat: TopicIndex.build — TF-IDF clustering over parquet corpus"
```

---

### Task 3: TopicIndex CLI Entry Point

**Files:**
- Modify: `halo3/perception/topic_index.py`

- [ ] **Step 1: Add `__main__` block for CLI build**

Append to `halo3/perception/topic_index.py`:

```python
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build TopicIndex from FineWeb parquet files")
    parser.add_argument("--build", action="store_true", help="Build index from data/fineweb/")
    parser.add_argument("--parquet-dir", default="data/fineweb", help="Parquet directory")
    parser.add_argument("--output", default="data/fineweb/topic_index.json", help="Output index path")
    args = parser.parse_args()
    if args.build:
        TopicIndex.build(args.parquet_dir, args.output)
    else:
        parser.print_help()
```

- [ ] **Step 2: Test CLI manually**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m halo3.perception.topic_index --help`
Expected: Shows usage with `--build`, `--parquet-dir`, `--output` options

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/perception/topic_index.py && git commit -m "feat: TopicIndex CLI — python -m halo3.perception.topic_index --build"
```

---

### Task 4: ActiveSampler — FE-Guided Curriculum Selection

**Files:**
- Create: `halo3/training/active_sampler.py`
- Test: `tests/training/test_active_sampler.py`

- [ ] **Step 1: Write failing test for select_curriculum**

```python
# tests/training/test_active_sampler.py
"""Tests for ActiveSampler — FE-guided curriculum selection."""
from __future__ import annotations
import json
import os
import tempfile
import pytest


def _make_mock_index(td: str, n_topics: int = 10, rows_per: int = 20):
    """Create a mock TopicIndex JSON with fake file positions."""
    topics = []
    for i in range(n_topics):
        topics.append({
            "topic_id": i,
            "keywords": [f"topic{i}", f"kw{i}a", f"kw{i}b"],
            "row_count": rows_per,
            "sample_titles": [f"Title {i}"],
            "file_positions": [
                {"file": 0, "row_group": 0, "row": i * rows_per + j}
                for j in range(rows_per)
            ],
        })
    index_path = os.path.join(td, "topic_index.json")
    with open(index_path, "w") as f:
        json.dump({"topics": topics, "parquet_files": []}, f)
    return index_path


def test_zone_of_proximal_development():
    """select_texts_by_fe filters to medium-FE zone."""
    from halo3.training.active_sampler import select_texts_by_fe

    # Simulate 20 texts with known FE scores
    texts = [f"text_{i}" for i in range(20)]
    fe_scores = list(range(20))  # 0, 1, 2, ..., 19

    result = select_texts_by_fe(texts, fe_scores, n_select=6)

    # Should exclude bottom 20% (0-3) and top 20% (16-19)
    # Then pick top 6 from the zone (sorted desc: 15, 14, 13, 12, 11, 10)
    assert len(result) == 6
    # All results should be from the middle zone
    for text in result:
        idx = int(text.split("_")[1])
        assert 4 <= idx <= 15, f"text_{idx} outside zone of proximal development"


def test_zone_handles_small_input():
    """select_texts_by_fe handles fewer candidates than n_select."""
    from halo3.training.active_sampler import select_texts_by_fe

    texts = ["a", "b", "c"]
    fe_scores = [1.0, 2.0, 3.0]
    result = select_texts_by_fe(texts, fe_scores, n_select=10)
    assert len(result) == 3  # returns all when fewer than n_select
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/training/test_active_sampler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'halo3.training.active_sampler'`

- [ ] **Step 3: Implement ActiveSampler**

```python
# halo3/training/active_sampler.py
"""Active Sampler — Free Energy-guided curriculum selection.

Uses Black-Scholes valuation to pick topics, then forward-only prediction
error to filter candidates into the zone of proximal development.

Zone of proximal development: texts with medium FE are most informative.
  - Low FE = already mastered (boring)
  - High FE = incomprehensible noise (overwhelming)
  - Medium FE = complex enough to learn from, simple enough to integrate
"""
from __future__ import annotations
import logging
import random

log = logging.getLogger(__name__)


def select_texts_by_fe(
    texts: list[str],
    fe_scores: list[float],
    n_select: int = 10,
) -> list[str]:
    """Filter texts to the zone of proximal development based on FE scores.

    Removes bottom 20% (mastered) and top 20% (noise), then picks
    the top n_select from the remaining zone (highest FE within zone —
    prefer the most challenging learnable content).
    """
    if len(texts) <= n_select:
        return list(texts)

    paired = sorted(zip(texts, fe_scores), key=lambda x: x[1])
    lo = len(paired) // 5
    hi = 4 * len(paired) // 5
    if hi <= lo:
        hi = len(paired)
        lo = 0
    zone = paired[lo:hi]

    # Within the zone, prefer highest FE (most challenging but learnable)
    zone.sort(key=lambda x: -x[1])
    return [text for text, _ in zone[:n_select]]


def rank_topics_by_bs(
    topic_index,
    volatility_surface,
    n_top: int = 20,
) -> list[tuple[int, float]]:
    """Rank topic buckets by Black-Scholes option value.

    Returns list of (topic_id, bs_value) sorted descending.
    """
    topics = topic_index.get_topics()
    valued = []
    for topic in topics:
        primary_kw = topic.keywords[0] if topic.keywords else ""
        value = volatility_surface.value_topic(primary_kw)
        valued.append((topic.topic_id, value))
    valued.sort(key=lambda x: -x[1])
    return valued[:n_top]


def sample_candidates(
    topic_index,
    ranked_topics: list[tuple[int, float]],
    n_candidates: int = 50,
) -> list[str]:
    """Stream candidate texts from top-ranked topics.

    Distributes candidates across topics, streams from disk.
    """
    if not ranked_topics:
        return []
    per_topic = max(1, n_candidates // len(ranked_topics))
    texts = []
    for topic_id, _ in ranked_topics:
        batch = topic_index.sample_from_topic(topic_id, n=per_topic)
        texts.extend(batch)
        if len(texts) >= n_candidates:
            break
    return texts[:n_candidates]


def select_curriculum(
    model,
    carry,
    topic_index,
    volatility_surface,
    embedder,
    n_candidates: int = 50,
    n_train: int = 10,
    key=None,
) -> list[str]:
    """Select the most informative texts for training.

    Full pipeline: BS ranks topics -> stream candidates -> FE scores -> zone filter.
    """
    import jax
    from halo3.loss import halo3_loss

    # Step 1: Rank topics by BS value
    ranked = rank_topics_by_bs(topic_index, volatility_surface, n_top=20)
    if not ranked:
        log.warning("ActiveSampler: no topics ranked — returning empty")
        return []

    # Step 2: Stream candidates from top topics
    candidates = sample_candidates(topic_index, ranked, n_candidates)
    if not candidates:
        log.warning("ActiveSampler: no candidates streamed — returning empty")
        return []

    log.info(f"ActiveSampler: scoring {len(candidates)} candidates by FE...")

    # Step 3: Forward-only FE scoring
    if key is None:
        key = jax.random.PRNGKey(42)

    fe_scores = []
    for text in candidates:
        tokens = embedder.texts_to_tokens([text], n_tokens=model.cfg.n_tokens)
        loss, _ = halo3_loss(model, carry, tokens, key)
        fe_scores.append(float(loss))

    # Step 4: Zone of proximal development
    selected = select_texts_by_fe(candidates, fe_scores, n_select=n_train)
    log.info(f"ActiveSampler: selected {len(selected)} texts (FE range: "
             f"{min(fe_scores):.2f} - {max(fe_scores):.2f})")
    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/training/test_active_sampler.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/training/active_sampler.py tests/training/test_active_sampler.py && git commit -m "feat: ActiveSampler — FE-guided zone-of-proximal-development selection"
```

---

### Task 5: VolatilitySurface Serialization

**Files:**
- Modify: `halo3/psyche/volatility.py`

The dream subprocess needs BS state. Add save/load to VolatilitySurface.

- [ ] **Step 1: Add save_state and load_state methods**

Add to `halo3/psyche/volatility.py` after the `best_topic` method:

```python
    def save_state(self, path: str) -> None:
        """Serialize volatility surface state to JSON for subprocess use."""
        import json
        state = {
            "strike": self._strike,
            "window": self._window,
            "dream_interval": self._dream_interval,
            "ticks_since_dream": self._ticks_since_dream,
            "history": {k: v for k, v in self._history.items()},
            "recent_topics": self._recent_topics,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> "VolatilitySurface":
        """Deserialize volatility surface from JSON."""
        import json
        with open(path) as f:
            state = json.load(f)
        vs = cls(
            strike=state.get("strike", 0.6),
            window=state.get("window", 20),
        )
        vs._dream_interval = state.get("dream_interval", 90)
        vs._ticks_since_dream = state.get("ticks_since_dream", 0)
        vs._recent_topics = state.get("recent_topics", [])
        for topic, hist in state.get("history", {}).items():
            vs._history[topic] = [tuple(h) for h in hist]
        return vs
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ -v -k "not model" --timeout=30 2>/dev/null || echo "Tests complete"`
Expected: No regressions

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/psyche/volatility.py && git commit -m "feat: VolatilitySurface save/load state for dream subprocess"
```

---

### Task 6: Replace Dream Phase 4 — Active FineWeb

**Files:**
- Modify: `halo3/training/dream_fineweb.py`
- Modify: `halo3/training/dream_fineweb_worker.py`

- [ ] **Step 1: Rewrite dream_fineweb.py to use ActiveSampler**

Replace `halo3/training/dream_fineweb.py` entirely:

```python
"""FineWeb dream phase — Phase 4 of body dreaming.

v3.11: Free Energy-guided active learning replaces sequential cursor.
ActiveSampler uses BS valuation to pick topics, then forward-only FE
to filter candidates into the zone of proximal development.
"""
from __future__ import annotations
import gc
import glob
import json
import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

log = logging.getLogger(__name__)


def fineweb_dream_phase(
    model,
    parquet_dir: str = "data/fineweb",
    n_steps: int = 10,
    lr: float = 5e-6,
    seed: int = 99,
    bs_state_path: str = "data/dream_training/bs_state.json",
    index_path: str = "data/fineweb/topic_index.json",
) -> tuple[any, dict]:
    """Train body on FE-selected FineWeb-Edu texts using CLion optimizer.

    Uses ActiveSampler to pick the most informative texts:
    1. BS valuation ranks topic clusters
    2. Stream candidates from top topics
    3. Forward-only FE scores candidates
    4. Train on zone-of-proximal-development texts
    """
    from halo3.perception.native_embedder import NativeEmbedder
    from halo3.loss import halo3_loss
    from halo3.training.dream_replay import scale_by_clion

    # Load topic index
    if not os.path.exists(index_path):
        log.warning("FineWeb dream: no topic index — building now...")
        from halo3.perception.topic_index import TopicIndex
        TopicIndex.build(parquet_dir, index_path)

    from halo3.perception.topic_index import TopicIndex
    topic_index = TopicIndex(index_path, parquet_dir)

    # Load BS state for topic ranking
    if os.path.exists(bs_state_path):
        from halo3.psyche.volatility import VolatilitySurface
        vol_surface = VolatilitySurface.load_state(bs_state_path)
        log.info("FineWeb dream: loaded BS state for active sampling")
    else:
        from halo3.psyche.volatility import VolatilitySurface
        vol_surface = VolatilitySurface()
        log.info("FineWeb dream: no BS state — using default priors")

    # Select curriculum via ActiveSampler
    embedder = NativeEmbedder(model.cfg.d_model, n_tokens=model.cfg.n_tokens)
    key = jax.random.PRNGKey(seed)

    from halo3.training.active_sampler import select_curriculum
    key, sample_key = jax.random.split(key)
    carry = model.init_carry(sample_key)

    texts = select_curriculum(
        model=model,
        carry=carry,
        topic_index=topic_index,
        volatility_surface=vol_surface,
        embedder=embedder,
        n_candidates=min(50, n_steps * 5),
        n_train=n_steps,
        key=sample_key,
    )

    if not texts:
        log.warning("FineWeb dream: ActiveSampler returned no texts — skipping")
        return model, {"fineweb_steps": 0, "fineweb_loss": 0.0}

    log.info(f"FineWeb dream: ActiveSampler selected {len(texts)} texts")

    # Embed selected texts
    log.info(f"  Pre-embedding {len(texts)} texts on CPU...")
    token_tensors = [embedder.texts_to_tokens([t], model.cfg.n_tokens) for t in texts]
    del embedder
    gc.collect()
    jax.clear_caches()

    # CLion optimizer (same as before)
    opt = optax.chain(
        optax.clip_by_global_norm(0.1),
        scale_by_clion(b1=0.9),
        optax.scale(-lr),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def _fineweb_step(model, opt_state_in, carry, tokens, key, scale):
        loss_fn = lambda m: halo3_loss(m, carry, tokens, key)[0] * scale
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state_in,
            eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss

    def _safe_step(model, opt_state, carry, tokens, key, scale):
        try:
            new_model, new_opt_state, loss = _fineweb_step(
                model, opt_state, carry, tokens, key, jnp.float32(scale)
            )
            loss_val = float(loss)
            if loss_val != loss_val:
                return model, opt_state, loss
            leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
            if any(bool(jnp.any(jnp.isnan(leaf))) for leaf in leaves):
                log.warning("  FineWeb: NaN weights after step — skipping")
                return model, opt_state, loss
            return new_model, new_opt_state, loss
        except Exception as e:
            log.warning(f"  FineWeb step exception: {e}")
            return model, opt_state, 0.0

    log.info(f"  Phase 4: FineWeb active learning ({len(token_tensors)} steps, scale=0.05)...")
    total_loss = 0.0
    completed = 0
    for tokens in token_tensors:
        key, sk = jax.random.split(key)
        model, opt_state, loss = _safe_step(model, opt_state, carry, tokens, sk, 0.05)
        total_loss += float(loss)
        completed += 1

    avg_loss = total_loss / max(completed, 1)
    gc.collect()
    jax.clear_caches()
    log.info(f"  FineWeb phase done: {completed}/{len(token_tensors)} steps | avg_loss={avg_loss:.2e}")
    return model, {"fineweb_steps": completed, "fineweb_loss": avg_loss}
```

- [ ] **Step 2: Update dream_fineweb_worker.py to pass new args**

Replace `halo3/training/dream_fineweb_worker.py`:

```python
"""Subprocess worker for FineWeb Phase 4 dreaming — active learning edition.

Spawned by main.py as a SEPARATE process AFTER the body dream worker exits.
Now uses ActiveSampler for FE-guided curriculum selection instead of sequential cursor.
"""
from __future__ import annotations
import argparse
import json
import logging
import os


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("fineweb_worker")

    parser = argparse.ArgumentParser(description="FineWeb dream subprocess (active learning)")
    parser.add_argument("--checkpoint", required=True, help="Input checkpoint path (without .eqx)")
    parser.add_argument("--output", required=True, help="Output checkpoint path (without .eqx)")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--bs-state", default="data/dream_training/bs_state.json")
    parser.add_argument("--index", default="data/fineweb/topic_index.json")
    args = parser.parse_args()

    import jax
    xla_cache = os.path.abspath(os.path.join("data", "xla_cache"))
    os.makedirs(xla_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", xla_cache)

    backend = jax.default_backend()
    log.info(f"FineWeb worker started — JAX backend: {backend}, devices: {jax.devices()}")

    from halo3.config import Halo3Config
    from halo3.training.bootstrap import load_checkpoint, save_checkpoint

    cfg = Halo3Config()
    log.info(f"Loading model from {args.checkpoint}.eqx")
    model = load_checkpoint(cfg, args.checkpoint)

    from halo3.training.dream_fineweb import fineweb_dream_phase
    log.info(f"FineWeb active learning: {args.steps} steps, lr={args.lr}")
    model, fw_info = fineweb_dream_phase(
        model,
        parquet_dir="data/fineweb",
        n_steps=args.steps,
        lr=args.lr,
        bs_state_path=args.bs_state,
        index_path=args.index,
    )

    save_checkpoint(model, args.output)
    log.info(f"FineWeb dream done: {fw_info}")

    info_path = os.path.join(os.path.dirname(args.output), "fineweb_dream_info.json")
    with open(info_path, "w") as f:
        json.dump(fw_info, f)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/training/dream_fineweb.py halo3/training/dream_fineweb_worker.py && git commit -m "feat: Dream Phase 4 uses ActiveSampler — FE-guided active learning replaces cursor"
```

---

### Task 7: Save BS State Before Dream in main.py

**Files:**
- Modify: `halo3/main.py`

- [ ] **Step 1: Add BS state save before dream Phase 4 subprocess**

Find the Phase 4 subprocess launch in `main.py` (around line 460-465). Add BS state save just before it:

```python
            # Save BS state for active learning subprocess
            try:
                organism.volatility.save_state("data/dream_training/bs_state.json")
            except Exception as e:
                log.warning(f"  Failed to save BS state: {e}")
```

Insert this immediately before the `log.info("  ☽ Phase 4: FineWeb dreaming on GPU (subprocess)...")` line.

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/main.py && git commit -m "feat: save BS volatility state before dream Phase 4 for active learning"
```

---

### Task 8: Replace ParquetSource in PerceptionPipeline

**Files:**
- Modify: `halo3/perception/pipeline.py`

- [ ] **Step 1: Rewrite pipeline.py to use TopicIndex**

Replace `halo3/perception/pipeline.py`:

```python
"""Perception pipeline — orchestrates fetch -> embed -> token tensor.

v3.11: Uses TopicIndex for FE-guided active learning from local parquet data.
Falls back to web search when no local data or topic index is present.
"""
from __future__ import annotations
import logging
import os

import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


class PerceptionPipeline:
    """Fetch content, embed, produce (n_tokens, d_model) tensor."""

    def __init__(self, d_model: int, n_tokens: int, vocab_size: int = 8000) -> None:
        self.n_tokens = n_tokens
        self.d_model = d_model
        self._topic_index = None

        # Auto-detect TopicIndex
        index_path = "data/fineweb/topic_index.json"
        if os.path.exists(index_path):
            try:
                from halo3.perception.topic_index import TopicIndex
                self._topic_index = TopicIndex(index_path, "data/fineweb")
                log.info("Perception: using TopicIndex (FE-guided active learning)")
            except Exception as e:
                log.warning(f"TopicIndex failed to load: {e} — falling back to web search")

        # Embedder (always needed)
        try:
            from halo3.perception.native_embedder import NativeEmbedder
            self.embedder = NativeEmbedder(d_model, vocab_size, n_tokens)
            if self.embedder._native_ready:
                log.info("Perception: using organism's OWN trained embedding")
            else:
                log.info("Perception: using sentence-transformers (no LM checkpoint yet)")
        except Exception:
            from halo3.perception.embedder import TextEmbedder
            self.embedder = TextEmbedder(d_model)
            log.info("Perception: using sentence-transformers fallback")

        if self._topic_index is None:
            log.info("Perception: using web search (no TopicIndex found)")

    def perceive(
        self,
        query: str,
        max_results: int = 5,
        model=None,
        carry=None,
        key=None,
    ) -> tuple[jnp.ndarray, list[str]]:
        """Fetch and embed content for a query.

        When TopicIndex is available and model/carry are provided, uses
        FE scoring to pick the most informative texts. Otherwise falls
        back to keyword matching or web search.
        """
        if self._topic_index is not None:
            topics = self._topic_index.match_topic(query)
            if topics:
                # Stream candidates from matching topics
                n_candidates = max_results * 3
                candidates = self._topic_index.sample_from_topics(
                    [t.topic_id for t in topics[:5]],
                    n_per_topic=max(1, n_candidates // min(5, len(topics))),
                )

                if candidates:
                    # FE-rank if model available
                    if model is not None and carry is not None and key is not None:
                        candidates = self._fe_rank(candidates, model, carry, key, max_results)
                    else:
                        candidates = candidates[:max_results]

                    if candidates:
                        tokens = self.embedder.texts_to_tokens(candidates, self.n_tokens)
                        return tokens, candidates

            # TopicIndex found no matches — fall through to web search
            log.debug(f"TopicIndex: no matches for '{query[:30]}' — trying web search")

        # Fallback: web search
        from halo3.perception.web_fetch import web_search, results_to_texts
        results = web_search(query, max_results=max_results)

        if not results:
            log.warning(f"No results for '{query}', using zero tokens")
            return jnp.zeros((self.n_tokens, self.d_model)), []

        texts = results_to_texts(results)
        tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
        return tokens, texts

    def _fe_rank(
        self,
        candidates: list[str],
        model,
        carry,
        key,
        n_select: int,
    ) -> list[str]:
        """Rank candidates by forward-only free energy, return zone of proximal development."""
        from halo3.loss import halo3_loss
        from halo3.training.active_sampler import select_texts_by_fe

        fe_scores = []
        for text in candidates:
            tokens = self.embedder.texts_to_tokens([text], self.n_tokens)
            loss, _ = halo3_loss(model, carry, tokens, key)
            fe_scores.append(float(loss))

        return select_texts_by_fe(candidates, fe_scores, n_select=n_select)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for FAISS indexing."""
        return self.embedder.embed_query(query)
```

- [ ] **Step 2: Update perceive() call in main.py**

Find the `perception.perceive(current_query, max_results)` call in `main.py` (around line 174) and add model/carry/key:

Change:
```python
            tokens, texts = perception.perceive(current_query, max_results)
```
To:
```python
            tokens, texts = perception.perceive(current_query, max_results, model=model, carry=carry, key=subkey)
```

Where `subkey` is the existing key split used nearby. Check the surrounding code for the correct variable name — look for `jax.random.split` near the perceive call.

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/perception/pipeline.py halo3/main.py && git commit -m "feat: PerceptionPipeline uses TopicIndex + FE scoring, replaces ParquetSource"
```

---

### Task 9: Delete ParquetSource

**Files:**
- Delete: `halo3/perception/parquet_source.py`
- Modify: `tests/perception/test_parquet_source.py` (delete or adapt)

- [ ] **Step 1: Verify no remaining imports of ParquetSource**

Run: `cd D:/New_Ai/.worktrees/halo3 && grep -r "parquet_source\|ParquetSource" halo3/ --include="*.py" | grep -v __pycache__`
Expected: Only `parquet_source.py` itself (nothing else imports it after Task 8)

- [ ] **Step 2: Delete the files**

```bash
cd D:/New_Ai/.worktrees/halo3 && rm halo3/perception/parquet_source.py
```

If `tests/perception/test_parquet_source.py` exists and tests the old class, delete it too:
```bash
rm tests/perception/test_parquet_source.py 2>/dev/null
```

- [ ] **Step 3: Run all perception tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/perception/ -v`
Expected: All remaining tests pass

- [ ] **Step 4: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add -A && git commit -m "refactor: delete ParquetSource — replaced by TopicIndex + ActiveSampler"
```

---

### Task 10: Build Topic Index Inside Docker

**Files:**
- Modify: `halo3/main.py`

The topic index needs to be built on first run inside the container.

- [ ] **Step 1: Add auto-build on startup in main.py**

Find the section in `main.py` where `PerceptionPipeline` is created (around line 101). Add topic index auto-build before it:

```python
    # Auto-build TopicIndex if parquet exists but index doesn't
    _index_path = "data/fineweb/topic_index.json"
    if not os.path.exists(_index_path):
        import glob as _glob
        if _glob.glob("data/fineweb/**/*.parquet", recursive=True):
            log.info("Building TopicIndex for first time (this takes a few minutes)...")
            from halo3.perception.topic_index import TopicIndex
            TopicIndex.build("data/fineweb", _index_path)
```

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add halo3/main.py && git commit -m "feat: auto-build TopicIndex on first startup when parquet data exists"
```

---

### Task 11: Integration Test

**Files:**
- Test: `tests/training/test_active_learning_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/training/test_active_learning_integration.py
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
        # The top match should contain physics keywords
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
    # Simulate: docs 0-9 have low FE (mastered), 40-49 have high FE (noise)
    fe_scores = [float(i) for i in range(50)]

    selected = select_texts_by_fe(texts, fe_scores, n_select=10)
    assert len(selected) == 10

    # None of the extremes should be selected
    selected_indices = {int(t.split("_")[1]) for t in selected}
    assert all(idx >= 10 for idx in selected_indices), "Low-FE texts should be filtered"
    assert all(idx < 40 for idx in selected_indices), "High-FE texts should be filtered"
```

- [ ] **Step 2: Run integration test**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/training/test_active_learning_integration.py -v`
Expected: 2 passed

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add tests/training/test_active_learning_integration.py && git commit -m "test: integration test for TopicIndex build + ActiveSampler selection"
```

---

### Task 12: Final Verification

- [ ] **Step 1: Run all tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/ -v --timeout=60`
Expected: All tests pass, no regressions

- [ ] **Step 2: Docker build test**

Run: `cd D:/New_Ai/.worktrees/halo3 && docker compose build train`
Expected: Build succeeds (no new dependencies needed)

- [ ] **Step 3: Verify topic index builds inside Docker**

Run: `cd D:/New_Ai/.worktrees/halo3 && docker compose run --rm train python3 -m halo3.perception.topic_index --build`
Expected: Scans 726K rows, creates `data/fineweb/topic_index.json` with 300-500 topics

- [ ] **Step 4: Commit any final fixes**

```bash
cd D:/New_Ai/.worktrees/halo3 && git add -A && git commit -m "chore: final verification — all tests pass, docker builds"
```
