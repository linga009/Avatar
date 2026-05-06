"""FAISS (IndexFlatIP) + SQLite episodic memory store.

FAISS index: 256-dim L2-normalized query embeddings, cosine similarity.
SQLite: full Episode data, keyed by UUID.

On startup: if FAISS index file is missing or corrupt, rebuild from SQLite.
"""
from __future__ import annotations

import json
import logging
import os
import pickle

import faiss
import numpy as np
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

from halo_fep.memory.schema import Episode

log = logging.getLogger(__name__)

_EMBED_DIM = 256  # query embedding stored in FAISS

_METADATA = sa.MetaData()
_EPISODES = sa.Table(
    "episodes", _METADATA,
    sa.Column("id",                 sa.String,  primary_key=True),
    sa.Column("timestamp",          sa.Float,   nullable=False),
    sa.Column("query",              sa.Text,    nullable=False),
    sa.Column("tokens",             sa.LargeBinary, nullable=False),   # pickled ndarray
    sa.Column("swarm_mu",           sa.LargeBinary, nullable=False),
    sa.Column("free_energy",        sa.Float,   nullable=False),
    sa.Column("free_energy_delta",  sa.Float,   nullable=False),
    sa.Column("llm_output",         sa.Text,    nullable=True),
    sa.Column("topic_tags",         sa.Text,    nullable=False),       # JSON list
    sa.Column("query_embed",        sa.LargeBinary, nullable=False),   # (256,) float32
)


class EpisodeStore:
    def __init__(self, path: str, embed_dim: int = _EMBED_DIM) -> None:
        os.makedirs(path, exist_ok=True)
        self._path      = path
        self._embed_dim = embed_dim
        self._db_path   = os.path.join(path, "episodes.db")
        self._idx_path  = os.path.join(path, "faiss.index")

        # NullPool: connections are closed immediately after use — no pooling.
        # Required on Windows so SQLite file handles are released promptly.
        self._engine = sa.create_engine(
            f"sqlite:///{self._db_path}", poolclass=NullPool
        )
        _METADATA.create_all(self._engine)

        if os.path.exists(self._idx_path):
            try:
                self._index = faiss.read_index(self._idx_path)
            except Exception:
                log.warning("FAISS index corrupt — rebuilding from SQLite.")
                self._index = self._new_index()
                self.rebuild_index()
        else:
            self._index = self._new_index()

        # In-memory id list mirrors FAISS row order
        self._ids: list[str] = self._load_ids()

    def __del__(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass

    def _new_index(self) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(self._embed_dim)

    def _load_ids(self) -> list[str]:
        """Load episode IDs in insertion order from SQLite."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES.c.id).order_by(_EPISODES.c.timestamp)
            ).fetchall()
        return [r[0] for r in rows]

    def _embed_from_query(self, query: str) -> np.ndarray:
        """Use first 256 UTF-8 bytes as a deterministic stand-in for embedding.

        NOTE: Real usage passes embedder.embed_text(query) directly via add().
        This fallback is for rebuild_index() only.
        """
        raw = query.encode("utf-8")[:self._embed_dim]
        vec = np.frombuffer(raw.ljust(self._embed_dim, b"\x00"), dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)

    def add(self, episode: Episode, query_embed: np.ndarray | None = None) -> None:
        """Persist episode. query_embed: (256,) float32 L2-normalized for FAISS."""
        if query_embed is None:
            query_embed = self._embed_from_query(episode.query)
        query_embed = query_embed.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embed)

        with self._engine.begin() as conn:
            conn.execute(_EPISODES.insert().values(
                id               = episode.id,
                timestamp        = episode.timestamp,
                query            = episode.query,
                tokens           = pickle.dumps(episode.tokens),
                swarm_mu         = pickle.dumps(episode.swarm_mu),
                free_energy      = float(episode.free_energy),
                free_energy_delta= float(episode.free_energy_delta),
                llm_output       = episode.llm_output,
                topic_tags       = json.dumps(episode.topic_tags),
                query_embed      = query_embed.tobytes(),
            ))
        self._index.add(query_embed)
        self._ids.append(episode.id)
        faiss.write_index(self._index, self._idx_path)

    def update_llm_output(self, episode_id: str, llm_output: str) -> None:
        """Update the llm_output field for an existing episode."""
        with self._engine.begin() as conn:
            conn.execute(
                _EPISODES.update()
                .where(_EPISODES.c.id == episode_id)
                .values(llm_output=llm_output)
            )

    def retrieve(self, query_embed: np.ndarray, k: int = 5) -> list[Episode]:
        """Return top-k episodes by cosine similarity to query_embed."""
        if self._index.ntotal == 0:
            return []
        qv = query_embed.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)
        k   = min(k, self._index.ntotal)
        _, idxs = self._index.search(qv, k)
        ids = [self._ids[i] for i in idxs[0] if i >= 0]
        return self._load_by_ids(ids)

    def get_recent(self, n: int = 500) -> list[Episode]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).order_by(_EPISODES.c.timestamp.desc()).limit(n)
            ).fetchall()
        return [self._row_to_episode(r) for r in reversed(rows)]

    def get_high_confidence(self, min_delta: float = -0.05) -> list[Episode]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).where(_EPISODES.c.free_energy_delta < min_delta)
                .order_by(_EPISODES.c.timestamp)
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_prioritized(
        self,
        n: int,
        since_timestamp: float = 0.0,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> tuple[list["Episode"], np.ndarray]:
        """Return up to n episodes sampled proportional to |free_energy_delta|^alpha.

        Higher |delta_fe| = more surprising/informative = higher priority.

        Args:
            n: Maximum number of episodes to return.
            since_timestamp: Only consider episodes after this Unix timestamp.
            alpha: Priority exponent. 0 = uniform, 1 = full priority.
            beta: Importance-sampling correction exponent. 0 = no correction.

        Returns:
            (episodes, weights) — weights are IS corrections in [0, 1].
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES)
                .where(_EPISODES.c.timestamp >= since_timestamp)
                .order_by(_EPISODES.c.timestamp)
            ).fetchall()

        if not rows:
            return [], np.array([], dtype=np.float32)

        episodes = [self._row_to_episode(r) for r in rows]

        # Priority = |delta_fe|^alpha, clipped to avoid zeros
        priorities = np.array(
            [abs(ep.free_energy_delta) ** alpha for ep in episodes],
            dtype=np.float32,
        )
        priorities = np.clip(priorities, 1e-8, None)
        probs = priorities / priorities.sum()

        n_sample = min(n, len(episodes))
        indices = np.random.choice(len(episodes), size=n_sample, replace=False, p=probs)

        sampled = [episodes[i] for i in indices]
        sampled_probs = probs[indices]

        # IS weights: w_i = (1/(N*p_i))^beta, normalized to [0,1]
        N = len(episodes)
        raw_weights = (1.0 / (N * sampled_probs + 1e-8)) ** beta
        weights = (raw_weights / raw_weights.max()).astype(np.float32)

        return sampled, weights

    def rebuild_index(self) -> None:
        """Reconstruct FAISS index from SQLite (recovery path)."""
        self._index = self._new_index()
        self._ids   = []
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).order_by(_EPISODES.c.timestamp)
            ).fetchall()
        for r in rows:
            qv = np.frombuffer(r.query_embed, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(qv)
            self._index.add(qv)
            self._ids.append(r.id)
        faiss.write_index(self._index, self._idx_path)

    def _load_by_ids(self, ids: list[str]) -> list[Episode]:
        if not ids:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).where(_EPISODES.c.id.in_(ids))
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, r) -> Episode:
        ep = Episode(
            id               = r.id,
            timestamp        = r.timestamp,
            query            = r.query,
            tokens           = pickle.loads(r.tokens),
            swarm_mu         = pickle.loads(r.swarm_mu),
            free_energy      = r.free_energy,
            free_energy_delta= r.free_energy_delta,
            llm_output       = r.llm_output,
            topic_tags       = json.loads(r.topic_tags),
        )
        return ep
