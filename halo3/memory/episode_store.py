"""Episode store — SQLite + FAISS for persistence and semantic search.

v3.1 additions:
  - Dead query tracking: persistent table of queries that return zero results
  - Negative memory: episodes marked as failures for anti-learning
  - Query success rate tracking
"""
from __future__ import annotations
import json
import logging
import os
import sqlite3
import numpy as np

log = logging.getLogger(__name__)

_EMBED_DIM = 384


class EpisodeStore:
    """Persistent episode storage with semantic retrieval."""

    def __init__(self, db_path: str = "data/episodes/episodes.db") -> None:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self._embeds: list[np.ndarray] = []
        self._ids: list[int] = []
        log.info(f"Episode store opened at {db_path}")

    def _create_tables(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                order_param REAL,
                mode TEXT,
                finding TEXT,
                timestamp TEXT,
                free_energy_delta REAL
            )
        """)
        # Dead queries: queries that consistently return zero results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS dead_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT UNIQUE,
                fail_count INTEGER DEFAULT 1,
                first_seen TEXT DEFAULT (datetime('now')),
                last_seen TEXT DEFAULT (datetime('now'))
            )
        """)
        # Query success tracking for meta-cognition
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS query_stats (
                query TEXT PRIMARY KEY,
                attempts INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                avg_r REAL DEFAULT 0.0
            )
        """)
        self.conn.commit()

    def add(self, episode) -> None:
        """Add an episode to the store."""
        cursor = self.conn.execute(
            "INSERT INTO episodes (query, order_param, mode, finding, timestamp, free_energy_delta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (episode.query, episode.order_param, episode.mode,
             episode.finding, episode.timestamp, episode.free_energy_delta),
        )
        self.conn.commit()
        ep_id = cursor.lastrowid

        if episode.query_embed is not None:
            self._embeds.append(episode.query_embed)
            self._ids.append(ep_id)

        # Update query stats
        self._update_query_stats(
            episode.query, episode.order_param,
            had_results=episode.finding is not None or episode.order_param > 0.1,
        )

    def record_dead_query(self, query: str) -> None:
        """Record a query that returned zero results."""
        self.conn.execute(
            "INSERT INTO dead_queries (query) VALUES (?) "
            "ON CONFLICT(query) DO UPDATE SET "
            "fail_count = fail_count + 1, last_seen = datetime('now')",
            (query,),
        )
        self.conn.commit()

    def get_dead_queries(self, limit: int = 50) -> list[str]:
        """Get known dead queries, most frequent failures first."""
        rows = self.conn.execute(
            "SELECT query FROM dead_queries ORDER BY fail_count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r[0] for r in rows]

    def is_dead_query(self, query: str) -> bool:
        """Check if a query is known to be dead."""
        row = self.conn.execute(
            "SELECT fail_count FROM dead_queries WHERE query = ?",
            (query,),
        ).fetchone()
        return row is not None and row[0] >= 2

    def _update_query_stats(self, query: str, r: float, had_results: bool) -> None:
        """Track query success rate for meta-cognition."""
        row = self.conn.execute(
            "SELECT attempts, successes, avg_r FROM query_stats WHERE query = ?",
            (query,),
        ).fetchone()
        if row:
            attempts = row[0] + 1
            successes = row[1] + (1 if had_results else 0)
            avg_r = (row[2] * row[0] + r) / attempts
            self.conn.execute(
                "UPDATE query_stats SET attempts=?, successes=?, avg_r=? WHERE query=?",
                (attempts, successes, avg_r, query),
            )
        else:
            self.conn.execute(
                "INSERT INTO query_stats (query, attempts, successes, avg_r) VALUES (?, 1, ?, ?)",
                (query, 1 if had_results else 0, r),
            )
        self.conn.commit()

    def get_query_success_rate(self, query: str) -> float:
        """Get historical success rate for a query (0-1)."""
        row = self.conn.execute(
            "SELECT attempts, successes FROM query_stats WHERE query = ?",
            (query,),
        ).fetchone()
        if not row or row[0] == 0:
            return 0.5  # unknown
        return row[1] / row[0]

    def get_best_queries(self, limit: int = 10) -> list[dict]:
        """Get queries with highest success rate and r values."""
        rows = self.conn.execute(
            "SELECT query, attempts, successes, avg_r FROM query_stats "
            "WHERE attempts >= 2 ORDER BY avg_r DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"query": r[0], "attempts": r[1], "successes": r[2], "avg_r": r[3]}
            for r in rows
        ]

    def get_findings(self, limit: int = 20) -> list[dict]:
        """Get recent findings (episodes with non-null finding)."""
        rows = self.conn.execute(
            "SELECT query, order_param, finding, timestamp FROM episodes "
            "WHERE finding IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"query": r[0], "r": r[1], "finding": r[2], "time": r[3]}
            for r in rows
        ]

    def get_high_confidence(self, threshold: float = 0.6) -> list[dict]:
        """Get high-confidence episodes for nightly dreaming."""
        rows = self.conn.execute(
            "SELECT query, order_param, mode, timestamp FROM episodes "
            "WHERE order_param > ? ORDER BY id DESC LIMIT 100",
            (threshold,),
        ).fetchall()
        return [
            {"query": r[0], "r": r[1], "mode": r[2], "time": r[3]}
            for r in rows
        ]

    def retrieve_similar(self, query_embed: np.ndarray, k: int = 5) -> list[int]:
        """Find k most similar episodes by cosine similarity."""
        if not self._embeds:
            return []
        matrix = np.stack(self._embeds)  # (N, 384)
        query_norm = query_embed / (np.linalg.norm(query_embed) + 1e-8)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        sims = (matrix / norms) @ query_norm
        top_k = np.argsort(sims)[-k:][::-1]
        return [self._ids[i] for i in top_k]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def flush(self) -> None:
        self.conn.commit()
