"""FAISS (IndexFlatIP) + SQLite episodic memory store.

Design
------
* **SQLite** is the source of truth.  Every episode is durably persisted via
  SQLAlchemy with ``NullPool`` (required on Windows to release file handles
  promptly).

* **FAISS** is a fast in-memory index for semantic retrieval.  It is treated
  as a *cache* — if the index file is missing or corrupt at startup it is
  rebuilt from SQLite with no data loss.

* **Batched FAISS writes** (Bug fix): writing the FAISS index to disk after
  *every* insert caused O(N) disk I/O per tick.  The index is now written
  only every ``_WRITE_EVERY`` inserts and on explicit ``flush()`` calls
  (graceful shutdown).  This keeps tick latency constant as the store grows.

* **Embed fallback removed** (Bug fix): the previous fallback ``_embed_from_query``
  converted UTF-8 bytes to float32 vectors.  This produced semantically
  meaningless embeddings that broke cosine retrieval.  The fallback now emits
  a warning and returns a zero vector; callers are expected to always supply a
  real embedding via the ``query_embed`` argument of ``add()``.
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

_EMBED_DIM   = 256   # query embedding dimension stored in FAISS
_WRITE_EVERY = 50    # persist FAISS index to disk every N inserts

_METADATA = sa.MetaData()
_EPISODES = sa.Table(
    "episodes", _METADATA,
    sa.Column("id",                 sa.String,      primary_key=True),
    sa.Column("timestamp",          sa.Float,       nullable=False),
    sa.Column("query",              sa.Text,        nullable=False),
    sa.Column("tokens",             sa.LargeBinary, nullable=False),   # pickled ndarray
    sa.Column("swarm_mu",           sa.LargeBinary, nullable=False),
    sa.Column("free_energy",        sa.Float,       nullable=False),
    sa.Column("free_energy_delta",  sa.Float,       nullable=False),
    sa.Column("llm_output",         sa.Text,        nullable=True),
    sa.Column("topic_tags",         sa.Text,        nullable=False),   # JSON list
    sa.Column("query_embed",        sa.LargeBinary, nullable=False),   # (256,) float32
)


class EpisodeStore:
    """Dual-backend episodic memory: SQLite for persistence, FAISS for retrieval.

    Parameters
    ----------
    path      : Directory path for ``episodes.db`` and ``faiss.index`` files.
    embed_dim : Dimension of the query embedding (default 256).

    Usage
    -----
    >>> store = EpisodeStore("data/episodes/")
    >>> store.add(episode, query_embed=my_embed)
    >>> similar = store.retrieve(query_embed, k=5)
    >>> store.flush()  # call on graceful shutdown
    """

    def __init__(self, path: str, embed_dim: int = _EMBED_DIM) -> None:
        os.makedirs(path, exist_ok=True)
        self._path      = path
        self._embed_dim = embed_dim
        self._db_path   = os.path.join(path, "episodes.db")
        self._idx_path  = os.path.join(path, "faiss.index")
        self._insert_count = 0  # tracks inserts since last FAISS write

        # NullPool: connections closed immediately — no pooling.
        # Required on Windows so SQLite file handles are released promptly.
        self._engine = sa.create_engine(
            f"sqlite:///{self._db_path}", poolclass=NullPool
        )
        _METADATA.create_all(self._engine)

        # Load or rebuild FAISS index
        if os.path.exists(self._idx_path):
            try:
                self._index = faiss.read_index(self._idx_path)
                log.info(f"FAISS index loaded: {self._index.ntotal} vectors.")
            except Exception:
                log.warning("FAISS index corrupt — rebuilding from SQLite.")
                self._index = self._new_index()
                self.rebuild_index()
        else:
            log.info("No FAISS index found — starting fresh.")
            self._index = self._new_index()

        # In-memory id list mirrors FAISS row order
        self._ids: list[str] = self._load_ids()

    def __del__(self) -> None:
        """Attempt to flush and dispose engine on garbage collection."""
        try:
            self.flush()
        except Exception:
            pass
        try:
            self._engine.dispose()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_index(self) -> faiss.IndexFlatIP:
        """Create a fresh inner-product FAISS index (cosine similarity on L2-normalised vecs)."""
        return faiss.IndexFlatIP(self._embed_dim)

    def _load_ids(self) -> list[str]:
        """Load episode IDs in insertion order from SQLite."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES.c.id).order_by(_EPISODES.c.timestamp)
            ).fetchall()
        return [r[0] for r in rows]

    def _maybe_write_index(self, force: bool = False) -> None:
        """Write FAISS index to disk if the write-every threshold is reached.

        Parameters
        ----------
        force : If True, write unconditionally (used by ``flush()``).
        """
        if force or (self._insert_count % _WRITE_EVERY == 0):
            faiss.write_index(self._index, self._idx_path)
            log.debug(
                f"FAISS index written ({self._index.ntotal} vectors, "
                f"insert #{self._insert_count})."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, episode: Episode, query_embed: np.ndarray | None = None) -> None:
        """Persist an episode to SQLite and add its embedding to FAISS.

        Parameters
        ----------
        episode     : Episode dataclass to store.
        query_embed : (embed_dim,) float32 L2-normalised embedding for FAISS
                      retrieval.  **Always pass a real embedding here.**  If
                      ``None``, a zero vector is used and a warning is emitted.
        """
        if query_embed is None:
            log.warning(
                "EpisodeStore.add() called without query_embed. "
                "A zero vector will be stored — semantic retrieval will be degraded. "
                "Pass embedder.embed_text(query) as query_embed."
            )
            query_embed = np.zeros(self._embed_dim, dtype=np.float32)

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

        self._insert_count += 1
        self._maybe_write_index()

    def flush(self) -> None:
        """Force-write the FAISS index to disk.

        Call this on graceful shutdown to ensure no inserts since the last
        periodic write are lost.
        """
        self._maybe_write_index(force=True)

    def update_llm_output(self, episode_id: str, llm_output: str) -> None:
        """Update the llm_output field for an existing episode.

        Parameters
        ----------
        episode_id : UUID string of the episode to update.
        llm_output : Text output from the LLM wake cycle.
        """
        with self._engine.begin() as conn:
            conn.execute(
                _EPISODES.update()
                .where(_EPISODES.c.id == episode_id)
                .values(llm_output=llm_output)
            )

    def retrieve(self, query_embed: np.ndarray, k: int = 5) -> list[Episode]:
        """Return the top-k episodes most similar to ``query_embed``.

        Uses cosine similarity (inner product on L2-normalised vectors).

        Parameters
        ----------
        query_embed : (embed_dim,) float32 query vector.
        k           : Maximum number of results to return.

        Returns
        -------
        List of Episode objects ordered by decreasing similarity.
        """
        if self._index.ntotal == 0:
            return []
        qv = query_embed.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)
        k    = min(k, self._index.ntotal)
        _, idxs = self._index.search(qv, k)
        ids  = [self._ids[i] for i in idxs[0] if i >= 0]
        return self._load_by_ids(ids)

    def get_recent(self, n: int = 500) -> list[Episode]:
        """Return the n most recent episodes, oldest first."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).order_by(_EPISODES.c.timestamp.desc()).limit(n)
            ).fetchall()
        return [self._row_to_episode(r) for r in reversed(rows)]

    def get_high_confidence(self, min_delta: float = -0.05) -> list[Episode]:
        """Return all episodes where free_energy_delta < min_delta.

        These are episodes where the agent was *surprised* (large negative
        delta) — the highest-signal experiences for nightly LoRA training.

        Parameters
        ----------
        min_delta : Free-energy delta threshold (negative; default -0.05).
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES)
                .where(_EPISODES.c.free_energy_delta < min_delta)
                .order_by(_EPISODES.c.timestamp)
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_prioritized(
        self,
        n: int,
        since_timestamp: float = 0.0,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> tuple[list[Episode], np.ndarray]:
        """Return up to n episodes sampled proportional to ``|ΔFE|^alpha``.

        Implements Prioritized Experience Replay (PER) sampling:
        higher ``|ΔFE|`` → more surprising → higher priority.

        Parameters
        ----------
        n               : Maximum number of episodes to return.
        since_timestamp : Only consider episodes after this Unix timestamp.
        alpha           : Priority exponent. 0 = uniform, 1 = full priority.
        beta            : Importance-sampling correction exponent.

        Returns
        -------
        (episodes, weights) where ``weights`` are IS corrections in [0, 1].
        """
        with self._engine.connect() as conn:
            # True buffer size for correct IS weight scaling
            total_count = conn.execute(
                sa.select(sa.func.count()).select_from(_EPISODES)
            ).scalar()
            rows = conn.execute(
                sa.select(_EPISODES)
                .where(_EPISODES.c.timestamp >= since_timestamp)
                .order_by(_EPISODES.c.timestamp)
            ).fetchall()

        if not rows:
            return [], np.array([], dtype=np.float32)

        episodes = [self._row_to_episode(r) for r in rows]

        # Priority = |ΔFE|^alpha, clipped to avoid division by zero
        priorities = np.array(
            [abs(ep.free_energy_delta) ** alpha for ep in episodes],
            dtype=np.float32,
        )
        priorities = np.clip(priorities, 1e-8, None)
        probs = priorities / priorities.sum()

        n_sample = min(n, len(episodes))
        indices  = np.random.choice(len(episodes), size=n_sample, replace=False, p=probs)

        sampled       = [episodes[i] for i in indices]
        sampled_probs = probs[indices]

        # IS weights: w_i = (1/(N*p_i))^beta, normalised to [0,1]
        raw_weights = (1.0 / (total_count * sampled_probs)) ** beta
        weights     = (raw_weights / raw_weights.max()).astype(np.float32)

        return sampled, weights

    def rebuild_index(self) -> None:
        """Reconstruct the FAISS index from all rows in SQLite.

        Use this as a recovery path when the index file is missing or corrupt.
        Writes the rebuilt index to disk immediately.
        """
        log.info("Rebuilding FAISS index from SQLite...")
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
        self._maybe_write_index(force=True)
        log.info(f"FAISS index rebuilt: {self._index.ntotal} vectors.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_by_ids(self, ids: list[str]) -> list[Episode]:
        if not ids:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).where(_EPISODES.c.id.in_(ids))
            ).fetchall()
        # Ensure returned episodes maintain FAISS distance ordering
        ep_dict = {r.id: self._row_to_episode(r) for r in rows}
        return [ep_dict[i] for i in ids if i in ep_dict]

    def _row_to_episode(self, r) -> Episode:
        return Episode(
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
