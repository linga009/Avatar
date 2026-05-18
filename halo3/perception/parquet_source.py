"""ParquetSource — local FineWeb-Edu Parquet as perception source.

Replaces web_search() for per-tick organism perception.
Returns list[SearchResult] with identical interface to web_fetch.web_search().
"""
from __future__ import annotations
import glob
import logging
import os
import re
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from halo3.perception.web_fetch import SearchResult

log = logging.getLogger(__name__)

_MIN_SCORE = 3       # int_score threshold for quality filtering
_MIN_WORD_LEN = 4    # minimum word length for keyword index
_SNIPPET_LEN = 500   # chars to use for SearchResult.snippet


class ParquetSource:
    """Local FineWeb-Edu Parquet perception source.

    Loads all *.parquet files found recursively under parquet_dir at startup,
    filters by int_score >= 3, builds an inverted keyword index, and returns
    SearchResult lists on each search() call — same interface as web_search().
    """

    def __init__(self, parquet_dir: str) -> None:
        self._texts: list[str] = []
        self._urls: list[str] = []
        self._cursor: int = 0
        self._index: dict[str, list[int]] = defaultdict(list)

        files = sorted(glob.glob(
            os.path.join(parquet_dir, "**/*.parquet"), recursive=True
        ))
        if not files:
            raise FileNotFoundError(f"No Parquet files in {parquet_dir}")

        log.info(f"ParquetSource: loading {len(files)} shard(s) from {parquet_dir}")
        for path in files:
            try:
                table = pq.read_table(path, columns=["text", "url", "int_score"])
                d = table.to_pydict()
                before = len(self._texts)
                for text, url, score in zip(d["text"], d["url"], d["int_score"]):
                    if (score or 0) >= _MIN_SCORE and text and text.strip():
                        self._texts.append(text)
                        self._urls.append(url or "")
                log.info(f"  {os.path.basename(path)}: {len(self._texts) - before:,} rows kept")
            except Exception as e:
                log.warning(f"  Skipping {os.path.basename(path)}: {e}")

        log.info(f"ParquetSource: {len(self._texts):,} total rows (int_score>={_MIN_SCORE})")
        self._build_index()

    def _build_index(self) -> None:
        """Build inverted keyword index: word -> [row_indices]."""
        log.info("ParquetSource: building keyword index...")
        for idx, text in enumerate(self._texts):
            for word in re.findall(rf"[a-z]{{{_MIN_WORD_LEN},}}", text.lower()):
                self._index[word].append(idx)
            if idx > 0 and idx % 50_000 == 0:
                log.info(f"  indexed {idx:,} / {len(self._texts):,} rows")
        log.info(f"ParquetSource: index ready ({len(self._index):,} unique words)")

    def search(self, query: str, n: int = 5) -> list[SearchResult]:
        """Find rows matching query keywords. Falls back to sequential cursor."""
        words = re.findall(rf"[a-z]{{{_MIN_WORD_LEN},}}", query.lower())

        if words:
            scores: dict[int, int] = defaultdict(int)
            for word in words:
                for idx in self._index.get(word, []):
                    scores[idx] += 1
            if scores:
                top = sorted(scores, key=lambda i: scores[i], reverse=True)[:n]
                return [self._make_result(i) for i in top]

        # Sequential cursor fallback
        log.debug(f"ParquetSource: no keyword match for '{query[:30]}' — sequential cursor")
        total = len(self._texts)
        results = []
        for _ in range(n):
            results.append(self._make_result(self._cursor % total))
            self._cursor += 1
        return results

    def _make_result(self, idx: int) -> SearchResult:
        text = self._texts[idx]
        title = " ".join(text.split()[:8])
        snippet = text[:_SNIPPET_LEN]
        return SearchResult(title=title, snippet=snippet, url=self._urls[idx])

    def sample_texts(self, n: int) -> list[str]:
        """Return n randomly sampled texts for dream batch training."""
        indices = np.random.choice(
            len(self._texts), size=min(n, len(self._texts)), replace=False
        )
        return [self._texts[i] for i in indices]
