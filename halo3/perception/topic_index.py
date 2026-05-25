"""TopicIndex — clustered topic map over FineWeb-Edu parquet corpus.

Streams parquet row groups to build a lightweight topic index. At runtime,
the index is a JSON file (~5-10MB) with pointers back into parquet files.
Texts are never held in RAM — streamed from disk on demand.
"""
from __future__ import annotations
import glob
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
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

    # ------------------------------------------------------------------
    # Build index from parquet corpus
    # ------------------------------------------------------------------
    @staticmethod
    def build(parquet_dir: str, output_path: str) -> "TopicIndex":
        """Two-pass TF-IDF scan over parquet shards to build a topic index.

        Pass 1: compute IDF for all words across qualifying texts.
        Pass 2: extract top-5 TF-IDF keywords per text, cluster into buckets.
        """
        _word_re = re.compile(rf"[a-z][a-z0-9]{{{_MIN_WORD_LEN - 1},}}")

        # Discover parquet shards
        shards = sorted(glob.glob(os.path.join(parquet_dir, "**", "*.parquet"), recursive=True))
        if not shards:
            raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
        log.info(f"TopicIndex.build: found {len(shards)} parquet shards")

        # ---- Pass 1: document frequencies ----
        doc_freq: defaultdict[str, int] = defaultdict(int)
        total_docs = 0
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for rg_idx in range(pf.num_row_groups):
                batch = pf.read_row_group(rg_idx, columns=["text", "int_score"])
                scores = batch.column("int_score")
                texts = batch.column("text")
                for i in range(len(scores)):
                    if scores[i].as_py() < _MIN_SCORE:
                        continue
                    text = texts[i].as_py()
                    if not text:
                        continue
                    words = set(_word_re.findall(text.lower()))
                    if not words:
                        continue
                    total_docs += 1
                    for w in words:
                        doc_freq[w] += 1

        if total_docs == 0:
            raise ValueError("No qualifying documents found (all filtered by score)")

        # Compute IDF, filter out overly common words
        idf: dict[str, float] = {}
        for w, df in doc_freq.items():
            score = math.log(total_docs / df)
            if score >= 1.0:
                idf[w] = score

        log.info(f"TopicIndex.build pass 1 done: {total_docs} docs, {len(idf)} IDF terms")

        # ---- Pass 2: TF-IDF keywords -> cluster into buckets ----
        # Each bucket: {keywords: set, positions: list}
        buckets: list[dict] = []
        rel_paths = [os.path.relpath(s, parquet_dir) for s in shards]

        for shard_idx, shard in enumerate(shards):
            pf = pq.ParquetFile(shard)
            for rg_idx in range(pf.num_row_groups):
                batch = pf.read_row_group(rg_idx, columns=["text", "int_score"])
                scores = batch.column("int_score")
                texts = batch.column("text")
                for row_in_rg in range(len(scores)):
                    if scores[row_in_rg].as_py() < _MIN_SCORE:
                        continue
                    text = texts[row_in_rg].as_py()
                    if not text:
                        continue
                    # Compute TF-IDF for this doc
                    words_in_doc = _word_re.findall(text.lower())
                    if not words_in_doc:
                        continue
                    tf: defaultdict[str, int] = defaultdict(int)
                    for w in words_in_doc:
                        tf[w] += 1
                    doc_len = len(words_in_doc)
                    tfidf = {}
                    for w, count in tf.items():
                        if w in idf:
                            tfidf[w] = (count / doc_len) * idf[w]
                    if not tfidf:
                        continue
                    # Top-5 keywords
                    top5 = set(sorted(tfidf, key=lambda w: -tfidf[w])[:5])

                    pos = {"file": shard_idx, "row_group": rg_idx, "row": row_in_rg}

                    # Find best matching bucket (need 2+ shared keywords)
                    best_idx = -1
                    best_overlap = 1  # threshold: need > 1
                    for bi, bucket in enumerate(buckets):
                        overlap = len(top5 & bucket["keywords"])
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_idx = bi
                    if best_idx >= 0:
                        buckets[best_idx]["keywords"] |= top5
                        buckets[best_idx]["positions"].append(pos)
                    else:
                        buckets.append({"keywords": set(top5), "positions": [pos]})

        # ---- Merge small buckets (< 5 rows) into nearest neighbor ----
        merged = True
        while merged:
            merged = False
            small = [i for i, b in enumerate(buckets) if len(b["positions"]) < 5]
            if not small:
                break
            for si in reversed(small):
                if si >= len(buckets):
                    continue
                sb = buckets[si]
                best_target = -1
                best_overlap = 0
                for ti, tb in enumerate(buckets):
                    if ti == si:
                        continue
                    overlap = len(sb["keywords"] & tb["keywords"])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_target = ti
                if best_target >= 0:
                    buckets[best_target]["keywords"] |= sb["keywords"]
                    buckets[best_target]["positions"].extend(sb["positions"])
                    buckets.pop(si)
                    merged = True

        # ---- Serialize to JSON ----
        topics_json = []
        for i, bucket in enumerate(buckets):
            topics_json.append({
                "topic_id": i,
                "keywords": sorted(bucket["keywords"]),
                "row_count": len(bucket["positions"]),
                "sample_titles": [],
                "file_positions": bucket["positions"],
            })

        data = {
            "parquet_files": [p.replace("\\", "/") for p in rel_paths],
            "topics": topics_json,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f)

        log.info(f"TopicIndex.build: {len(topics_json)} topics, "
                 f"{sum(t['row_count'] for t in topics_json)} positions -> {output_path}")
        return TopicIndex(output_path, parquet_dir)

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
