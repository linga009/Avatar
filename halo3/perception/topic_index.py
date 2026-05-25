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
from dataclasses import dataclass

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
        # Build keyword -> topic index lookup for fast matching
        self._keyword_to_topics: dict[str, list[int]] = {}
        for i, topic in enumerate(self._topics):
            for kw in topic.keywords:
                self._keyword_to_topics.setdefault(kw, []).append(i)
        log.info(f"TopicIndex: loaded {len(self._topics)} topics from {index_path}")

    def get_topics(self) -> list[TopicBucket]:
        return self._topics

    def match_topic(self, query: str) -> list[TopicBucket]:
        """Find topics whose keywords overlap with query words."""
        words = set(re.findall(rf"[a-z][a-z0-9]{{{_MIN_WORD_LEN - 1},}}", query.lower()))
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
