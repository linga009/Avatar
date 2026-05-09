"""Episode store — SQLite + FAISS for persistence and semantic search."""
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
