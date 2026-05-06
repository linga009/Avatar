# Three-Phase Nightly Dream Cycle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single nightly LoRA session with a three-phase dream cycle (spaced repetition → curiosity gap-filling → scaffolded build) running in a configurable 60-minute nightly window.

**Architecture:** `NightlyDreamCycle` in `halo_fep/training/dream_cycle.py` orchestrates three phases using a new `LearningJournal` SQLite store. `EpisodeStore` gains auto-topic-tagging in `add()`, `get_curiosity_gaps()`, and a `topic_boost` parameter in `get_prioritized()`. All existing components (`LoRATrainer`, `topic_bootstrap`, `hyperbolic_pretrain`) are reused unchanged. `HeartbeatLoop` in `main.py` is wired to use `NightlyDreamCycle` when available, with fallback to the old `LoRATrainer` path.

**Tech Stack:** Python, SQLAlchemy (NullPool/SQLite), JAX/Equinox, optax, SM-2 spaced repetition

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `halo_fep/config.py` | Add 9 dream-cycle config fields |
| Create | `halo_fep/memory/learning_journal.py` | SM-2 forgetting curves + per-night topic history |
| Modify | `halo_fep/memory/episode_store.py` | Auto-tag on `add()`, `get_curiosity_gaps()`, `topic_boost` in `get_prioritized()` |
| Modify | `halo_fep/training/topic_bootstrap.py` | Add `iter_cluster_token_batches()` (single-cluster Wikipedia stream) |
| Create | `halo_fep/training/dream_cycle.py` | `NightlyDreamCycle` orchestrator |
| Modify | `halo_fep/main.py` | Wire `NightlyDreamCycle` into `HeartbeatLoop` |
| Modify | `halo_fep/tests/test_config.py` | New config field tests |
| Create | `halo_fep/memory/tests/test_learning_journal.py` | SM-2, due-episodes, record |
| Modify | `halo_fep/memory/tests/test_episode_store.py` | Auto-tag, curiosity gaps, topic_boost |
| Modify | `halo_fep/training/tests/test_topic_bootstrap.py` | iter_cluster_token_batches |
| Create | `halo_fep/training/tests/test_dream_cycle.py` | Orchestrator, phase skips, journal write |
| Modify | `halo_fep/tests/test_main.py` | Nightly window duration, dream_cycle wiring |

---

## Task 1: Config Extensions

**Files:**
- Modify: `halo_fep/config.py`
- Modify: `halo_fep/tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `halo_fep/tests/test_config.py`:

```python
def test_dream_cycle_config_defaults():
    cfg = HaloFEPConfig()
    assert cfg.nightly_duration_min == 60
    assert cfg.spaced_rep_intervals == (1, 3, 7, 14, 30)
    assert cfg.curiosity_fe_threshold == 0.5
    assert cfg.curiosity_min_encounters == 3
    assert cfg.curiosity_since_days == 7
    assert cfg.phase1_steps == 150
    assert cfg.phase2_steps_per_cluster == 200
    assert cfg.phase3_steps == 100
    assert cfg.scaffold_boost == 2.0


def test_nightly_duration_min_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        HaloFEPConfig(nightly_duration_min=0)
```

- [ ] **Step 2: Run to confirm failure**

```
pytest halo_fep/tests/test_config.py::test_dream_cycle_config_defaults halo_fep/tests/test_config.py::test_nightly_duration_min_must_be_positive -v
```

Expected: FAIL with `AttributeError: 'HaloFEPConfig' object has no attribute 'nightly_duration_min'`

- [ ] **Step 3: Add fields to `halo_fep/config.py`**

After the `# Heartbeat` block (line 57–59), add a new `# Nightly dream cycle` block, and add the validation in `__post_init__`. Full replacement for lines 57–69:

```python
    # Heartbeat
    wake_threshold: float = 2.5   # FE above this triggers LLM wake cycle
    tick_interval:  int   = 60    # seconds between subconscious ticks

    # Nightly dream cycle
    nightly_duration_min:     int   = 60              # total window length in minutes
    spaced_rep_intervals:     tuple = (1, 3, 7, 14, 30)  # SM-2 review intervals (days)
    curiosity_fe_threshold:   float = 0.5             # min FE to count as gap encounter
    curiosity_min_encounters: int   = 3               # min encounters to trigger Phase 2
    curiosity_since_days:     int   = 7               # lookback window for gap detection
    phase1_steps:             int   = 150             # SGD steps in Phase 1
    phase2_steps_per_cluster: int   = 200             # SGD steps per cluster in Phase 2
    phase3_steps:             int   = 100             # SGD steps in Phase 3
    scaffold_boost:           float = 2.0             # priority multiplier for yesterday's topics

    def __post_init__(self) -> None:
        if self.n_agents % self.coarse_k != 0:
            raise ValueError(f"n_agents ({self.n_agents}) must be divisible by coarse_k ({self.coarse_k})")
        if self.n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {self.n_tokens}")
        if self.wake_threshold <= 0.0:
            raise ValueError(f"wake_threshold must be > 0, got {self.wake_threshold}")
        if self.tick_interval <= 0:
            raise ValueError(f"tick_interval must be > 0, got {self.tick_interval}")
        if self.nightly_duration_min <= 0:
            raise ValueError(f"nightly_duration_min must be > 0, got {self.nightly_duration_min}")
```

- [ ] **Step 4: Run tests to confirm passing**

```
pytest halo_fep/tests/test_config.py -v
```

Expected: all config tests pass.

- [ ] **Step 5: Commit**

```bash
git add halo_fep/config.py halo_fep/tests/test_config.py
git commit -m "feat(config): add nine dream-cycle config fields"
```

---

## Task 2: LearningJournal

**Files:**
- Create: `halo_fep/memory/learning_journal.py`
- Create: `halo_fep/memory/tests/test_learning_journal.py`

- [ ] **Step 1: Write failing tests**

Create `halo_fep/memory/tests/test_learning_journal.py`:

```python
"""Unit tests for LearningJournal — SM-2 curves and per-night journal."""
import os
import tempfile

from halo_fep.memory.learning_journal import LearningJournal


def _make_journal(tmp_path: str, intervals=(1, 3, 7, 14, 30)):
    return LearningJournal(tmp_path, spaced_rep_intervals=intervals)


def test_enroll_and_get_due():
    """Episodes enrolled today are due tomorrow (intervals[0]=1 day)."""
    with tempfile.TemporaryDirectory() as d:
        j = _make_journal(d)
        j.enroll_episodes(["ep1", "ep2"], today="2026-05-06")
        # Not due today
        assert j.get_due_episodes(None, "2026-05-06") == []  # type: ignore[arg-type]
        # Due tomorrow
        due_ids = [
            ep_id
            for ep_id, state in j._forgetting_state.items()
            if state["next_due"] <= "2026-05-07"
        ]
        assert sorted(due_ids) == ["ep1", "ep2"]


def test_update_forgetting_state_success_advances_interval():
    """Successful review moves stability up and schedules next interval."""
    with tempfile.TemporaryDirectory() as d:
        j = _make_journal(d, intervals=(1, 3, 7))
        j.enroll_episodes(["ep1"], today="2026-05-06")
        assert j._forgetting_state["ep1"]["stability"] == 0
        j.update_forgetting_state(["ep1"], today="2026-05-07", success=True)
        assert j._forgetting_state["ep1"]["stability"] == 1
        assert j._forgetting_state["ep1"]["next_due"] == "2026-05-10"  # +3 days


def test_update_forgetting_state_failure_leaves_unchanged():
    """Failed review leaves stability and next_due unchanged."""
    with tempfile.TemporaryDirectory() as d:
        j = _make_journal(d, intervals=(1, 3, 7))
        j.enroll_episodes(["ep1"], today="2026-05-06")
        before = dict(j._forgetting_state["ep1"])
        j.update_forgetting_state(["ep1"], today="2026-05-07", success=False)
        assert j._forgetting_state["ep1"] == before


def test_get_yesterday_topics_empty_when_no_entries():
    with tempfile.TemporaryDirectory() as d:
        j = _make_journal(d)
        assert j.get_yesterday_topics("2026-05-06") == []


def test_record_and_get_yesterday_topics():
    """record() persists trained_topics; get_yesterday_topics reads them back."""
    with tempfile.TemporaryDirectory() as d:
        j = _make_journal(d)
        j.record("2026-05-05", [0, 3], {}, {}, {})
        topics = j.get_yesterday_topics("2026-05-06")
        assert topics == [0, 3]


def test_forgetting_state_persists_across_instances():
    """Forgetting state loaded from DB on construction."""
    with tempfile.TemporaryDirectory() as d:
        j1 = _make_journal(d)
        j1.enroll_episodes(["ep1"], today="2026-05-06")
        j1.record("2026-05-06", [], {}, {}, {})
        del j1
        j2 = _make_journal(d)
        assert "ep1" in j2._forgetting_state


def test_record_upserts_same_date():
    """Calling record() twice on same date overwrites the row."""
    with tempfile.TemporaryDirectory() as d:
        j = _make_journal(d)
        j.record("2026-05-05", [0], {}, {}, {})
        j.record("2026-05-05", [1, 2], {}, {}, {})
        topics = j.get_yesterday_topics("2026-05-06")
        assert topics == [1, 2]
```

- [ ] **Step 2: Run to confirm failure**

```
pytest halo_fep/memory/tests/test_learning_journal.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'halo_fep.memory.learning_journal'`

- [ ] **Step 3: Create `halo_fep/memory/learning_journal.py`**

```python
"""Per-night learning journal for the three-phase dream cycle.

Tracks SM-2 spaced-repetition forgetting curves per episode and records
which topic clusters were trained each night.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.pool import NullPool

if TYPE_CHECKING:
    from halo_fep.memory.episode_store import EpisodeStore
    from halo_fep.memory.schema import Episode

log = logging.getLogger(__name__)

_METADATA_J = sa.MetaData()
_JOURNAL = sa.Table(
    "learning_journal", _METADATA_J,
    sa.Column("date",               sa.String, primary_key=True),
    sa.Column("trained_topics",     sa.Text,   nullable=False),   # JSON list[int]
    sa.Column("phase1_loss_before", sa.Float,  nullable=True),
    sa.Column("phase1_loss_after",  sa.Float,  nullable=True),
    sa.Column("phase2_clusters",    sa.Text,   nullable=True),    # JSON list[int]
    sa.Column("phase3_loss_before", sa.Float,  nullable=True),
    sa.Column("phase3_loss_after",  sa.Float,  nullable=True),
    sa.Column("forgetting_state",   sa.Text,   nullable=False),   # JSON dict
)


class LearningJournal:
    """SQLite-backed journal of nightly dream cycle runs.

    Maintains an in-memory _forgetting_state dict mapping episode_id to
    {"stability": int, "next_due": "YYYY-MM-DD"}.  The dict is persisted to
    the forgetting_state column on every record() call and reloaded from the
    most recent row on construction.
    """

    def __init__(
        self,
        path: str,
        spaced_rep_intervals: tuple[int, ...] = (1, 3, 7, 14, 30),
    ) -> None:
        os.makedirs(path, exist_ok=True)
        self._db_path   = os.path.join(path, "journal.db")
        self._intervals = spaced_rep_intervals
        self._engine    = sa.create_engine(
            f"sqlite:///{self._db_path}", poolclass=NullPool
        )
        _METADATA_J.create_all(self._engine)
        self._forgetting_state: dict[str, dict] = self._load_forgetting_state()

    def __del__(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_forgetting_state(self) -> dict:
        """Load forgetting state from most recent journal row, or empty dict."""
        with self._engine.connect() as conn:
            row = conn.execute(
                sa.select(_JOURNAL.c.forgetting_state)
                .order_by(_JOURNAL.c.date.desc())
                .limit(1)
            ).fetchone()
        return json.loads(row.forgetting_state) if row else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_due_episodes(
        self,
        store: "EpisodeStore",
        today: str,
    ) -> list["Episode"]:
        """Return episodes whose spaced-repetition interval has elapsed.

        Queries store._load_by_ids for the matching episode IDs.
        Returns [] if store is None or no episodes are due.
        """
        if store is None:
            return []
        due_ids = [
            ep_id
            for ep_id, state in self._forgetting_state.items()
            if state["next_due"] <= today
        ]
        if not due_ids:
            return []
        return store._load_by_ids(due_ids)

    def update_forgetting_state(
        self,
        episode_ids: list[str],
        today: str,
        success: bool,
    ) -> None:
        """Advance SM-2 interval on success; leave unchanged on failure.

        On success: stability += 1, next_due = today + intervals[stability].
        On failure: no change — the same interval is retried tomorrow.
        """
        for ep_id in episode_ids:
            state = self._forgetting_state.get(
                ep_id, {"stability": 0, "next_due": today}
            )
            if success:
                new_stability = state["stability"] + 1
                idx           = min(new_stability, len(self._intervals) - 1)
                days          = self._intervals[idx]
                next_due      = (
                    datetime.date.fromisoformat(today)
                    + datetime.timedelta(days=days)
                ).isoformat()
                self._forgetting_state[ep_id] = {
                    "stability": new_stability,
                    "next_due":  next_due,
                }

    def enroll_episodes(self, episode_ids: list[str], today: str) -> None:
        """Enroll new episodes; first review scheduled intervals[0] days from today."""
        first_due = (
            datetime.date.fromisoformat(today)
            + datetime.timedelta(days=self._intervals[0])
        ).isoformat()
        for ep_id in episode_ids:
            if ep_id not in self._forgetting_state:
                self._forgetting_state[ep_id] = {
                    "stability": 0,
                    "next_due":  first_due,
                }

    def get_yesterday_topics(self, today: str) -> list[int]:
        """Return trained_topics from the most recent entry before today."""
        with self._engine.connect() as conn:
            row = conn.execute(
                sa.select(_JOURNAL.c.trained_topics)
                .where(_JOURNAL.c.date < today)
                .order_by(_JOURNAL.c.date.desc())
                .limit(1)
            ).fetchone()
        return json.loads(row.trained_topics) if row else []

    def record(
        self,
        date: str,
        trained_topics: list[int],
        log1: dict,
        log2: dict,
        log3: dict,
    ) -> None:
        """Upsert one journal row and persist current forgetting state."""
        row_data = {
            "trained_topics":     json.dumps(trained_topics),
            "phase1_loss_before": log1.get("loss_before"),
            "phase1_loss_after":  log1.get("loss_after"),
            "phase2_clusters":    json.dumps(log2.get("clusters_trained", [])),
            "phase3_loss_before": log3.get("loss_before"),
            "phase3_loss_after":  log3.get("loss_after"),
            "forgetting_state":   json.dumps(self._forgetting_state),
        }
        with self._engine.begin() as conn:
            existing = conn.execute(
                sa.select(_JOURNAL).where(_JOURNAL.c.date == date)
            ).fetchone()
            if existing:
                conn.execute(
                    _JOURNAL.update()
                    .where(_JOURNAL.c.date == date)
                    .values(**row_data)
                )
            else:
                conn.execute(_JOURNAL.insert().values(date=date, **row_data))
```

- [ ] **Step 4: Run tests to confirm passing**

```
pytest halo_fep/memory/tests/test_learning_journal.py -v
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add halo_fep/memory/learning_journal.py halo_fep/memory/tests/test_learning_journal.py
git commit -m "feat(memory): LearningJournal — SM-2 forgetting curves + nightly topic history"
```

---

## Task 3: EpisodeStore Additions

**Files:**
- Modify: `halo_fep/memory/episode_store.py`
- Modify: `halo_fep/memory/tests/test_episode_store.py`

### Background

`episode_store.py` already has a `topic_tags` column (JSON list). The `Episode` dataclass already has `topic_tags: list[str]`. We need to:
1. Auto-populate `topic_tags` in `add()` when the list is empty, by matching the query against `TOPIC_KEYWORDS` cluster keywords.
2. Add `get_curiosity_gaps()` that counts high-FE episodes per cluster.
3. Add `topic_boost: dict[str, float] | None = None` to `get_prioritized()`.

- [ ] **Step 1: Write failing tests**

Append to `halo_fep/memory/tests/test_episode_store.py`:

```python
import datetime as _dt
import time as _time


def _make_episode_with_query(query: str, fe: float = 1.5, delta: float = -0.1) -> "Episode":
    import numpy as np
    from halo_fep.memory.schema import Episode
    return Episode(
        query=query,
        tokens=np.zeros((2, 256), dtype=np.float32),
        swarm_mu=np.zeros((256, 8), dtype=np.float32),
        free_energy=fe,
        free_energy_delta=delta,
    )


def test_add_auto_tags_topic_from_query():
    """add() sets topic_tags based on query keywords if tags are empty."""
    import tempfile, numpy as np
    from halo_fep.memory.episode_store import EpisodeStore
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(d)
        ep = _make_episode_with_query("I wrote a new algorithm for sorting")
        store.add(ep, query_embed=np.zeros(256, dtype=np.float32))
        loaded = store._load_by_ids([ep.id])[0]
        # cluster 1 keywords include "algorithm"
        assert "1" in loaded.topic_tags


def test_add_preserves_existing_tags():
    """add() does not overwrite tags that are already set."""
    import tempfile, numpy as np
    from halo_fep.memory.episode_store import EpisodeStore
    from halo_fep.memory.schema import Episode
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(d)
        ep = _make_episode_with_query("algorithm")
        ep.topic_tags = ["99"]   # manually set
        store.add(ep, query_embed=np.zeros(256, dtype=np.float32))
        loaded = store._load_by_ids([ep.id])[0]
        assert loaded.topic_tags == ["99"]


def test_add_falls_back_to_general_when_no_match():
    import tempfile, numpy as np
    from halo_fep.memory.episode_store import EpisodeStore
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(d)
        ep = _make_episode_with_query("xyzzy plugh")
        store.add(ep, query_embed=np.zeros(256, dtype=np.float32))
        loaded = store._load_by_ids([ep.id])[0]
        assert loaded.topic_tags == ["general"]


def test_get_curiosity_gaps_returns_clusters_with_enough_high_fe_episodes():
    """Clusters with >= min_encounters high-FE episodes in window are returned."""
    import tempfile, numpy as np
    from halo_fep.memory.episode_store import EpisodeStore
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(d)
        # 4 algorithm episodes with high FE
        for i in range(4):
            ep = _make_episode_with_query("algorithm system", fe=1.5)
            store.add(ep, query_embed=np.zeros(256, dtype=np.float32))
        # 1 history episode with high FE (below min_encounters=3)
        ep = _make_episode_with_query("ancient history civilization", fe=1.5)
        store.add(ep, query_embed=np.zeros(256, dtype=np.float32))

        gaps = store.get_curiosity_gaps(since_days=1, min_encounters=3, fe_threshold=1.0)
        assert "1" in [str(g) for g in gaps]   # cluster 1 = algorithm
        assert "6" not in [str(g) for g in gaps]  # cluster 6 = history, only 1 ep


def test_get_curiosity_gaps_excludes_general():
    """Episodes tagged 'general' are never returned as gaps."""
    import tempfile, numpy as np
    from halo_fep.memory.episode_store import EpisodeStore
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(d)
        for _ in range(5):
            ep = _make_episode_with_query("xyzzy plugh", fe=2.0)
            store.add(ep, query_embed=np.zeros(256, dtype=np.float32))
        gaps = store.get_curiosity_gaps(since_days=1, min_encounters=3, fe_threshold=1.0)
        assert gaps == []


def test_get_prioritized_topic_boost_increases_sampling_rate():
    """Episodes in boosted clusters appear more often when sampled."""
    import tempfile, numpy as np
    from halo_fep.memory.episode_store import EpisodeStore
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(d)
        # 10 algorithm episodes, 10 history episodes, all equal delta_fe
        for _ in range(10):
            ep = _make_episode_with_query("algorithm system", delta=-0.1)
            store.add(ep, query_embed=rng.randn(256).astype(np.float32))
        for _ in range(10):
            ep = _make_episode_with_query("ancient history", delta=-0.1)
            store.add(ep, query_embed=rng.randn(256).astype(np.float32))

        # Boost cluster "1" (algorithm) heavily
        episodes, _ = store.get_prioritized(
            n=20, topic_boost={"1": 100.0}
        )
        algo_count = sum(1 for e in episodes if "1" in e.topic_tags)
        # With a 100x boost, algorithm episodes should dominate
        assert algo_count > 12
```

- [ ] **Step 2: Run to confirm failure**

```
pytest halo_fep/memory/tests/test_episode_store.py::test_add_auto_tags_topic_from_query halo_fep/memory/tests/test_episode_store.py::test_get_curiosity_gaps_returns_clusters_with_enough_high_fe_episodes halo_fep/memory/tests/test_episode_store.py::test_get_prioritized_topic_boost_increases_sampling_rate -v
```

Expected: FAIL with `AttributeError` or `TypeError`

- [ ] **Step 3: Add `_match_topic_clusters()` helper to `episode_store.py`**

Add this function just before the `EpisodeStore` class definition (after the `log = ...` line at the top of the file):

```python
def _match_topic_clusters(query: str) -> list[str]:
    """Return cluster IDs (as strings) matching query keywords.

    Imports TOPIC_KEYWORDS at call time to avoid a circular import at
    module load. Returns ["general"] if no cluster matches.
    """
    from halo_fep.training.topic_bootstrap import TOPIC_KEYWORDS
    q = query.lower()
    matched = [
        str(cluster_id)
        for cluster_id, keywords in TOPIC_KEYWORDS.items()
        if any(kw in q for kw in keywords)
    ]
    return matched if matched else ["general"]
```

- [ ] **Step 4: Update `add()` to auto-tag**

Replace the `add()` method body. The only change is adding two lines before the `with self._engine.begin()` block — after `faiss.normalize_L2(query_embed)`:

```python
    def add(self, episode: Episode, query_embed: np.ndarray | None = None) -> None:
        """Persist episode. query_embed: (256,) float32 L2-normalized for FAISS."""
        if query_embed is None:
            query_embed = self._embed_from_query(episode.query)
        query_embed = query_embed.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embed)

        # Auto-tag topic cluster from query if not already set
        if not episode.topic_tags:
            episode.topic_tags = _match_topic_clusters(episode.query)

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
```

- [ ] **Step 5: Add `get_curiosity_gaps()` method**

Add after `get_high_confidence()` (after line 155 in the original file):

```python
    def get_curiosity_gaps(
        self,
        since_days: int = 7,
        min_encounters: int = 3,
        fe_threshold: float = 0.5,
    ) -> list[int]:
        """Return cluster IDs where the agent was repeatedly surprised.

        Groups episodes from the last `since_days` by their first topic tag.
        Returns cluster IDs (integers) with >= min_encounters episodes that
        had free_energy > fe_threshold, ordered by count descending.
        Excludes the "general" catch-all cluster.
        """
        import datetime as _dt
        since_ts = (
            _dt.datetime.now() - _dt.timedelta(days=since_days)
        ).timestamp()
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES.c.topic_tags, _EPISODES.c.free_energy)
                .where(_EPISODES.c.timestamp >= since_ts)
                .where(_EPISODES.c.free_energy > fe_threshold)
            ).fetchall()

        counts: dict[str, int] = {}
        for row in rows:
            tags = json.loads(row.topic_tags)
            if tags and tags[0] != "general":
                counts[tags[0]] = counts.get(tags[0], 0) + 1

        return [
            int(cluster_id)
            for cluster_id, count in sorted(counts.items(), key=lambda x: -x[1])
            if count >= min_encounters
        ]
```

- [ ] **Step 6: Add `topic_boost` parameter to `get_prioritized()`**

Replace the `get_prioritized()` signature and body (current lines 157–210):

```python
    def get_prioritized(
        self,
        n: int,
        since_timestamp: float = 0.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        topic_boost: dict[str, float] | None = None,
    ) -> tuple[list["Episode"], np.ndarray]:
        """Return up to n episodes sampled proportional to |free_energy_delta|^alpha.

        topic_boost: optional dict mapping tag string to priority multiplier.
                     Episodes whose first topic_tag is in topic_boost get their
                     raw priority multiplied by the corresponding factor before
                     normalization.
        """
        with self._engine.connect() as conn:
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

        priorities = np.array(
            [abs(ep.free_energy_delta) ** alpha for ep in episodes],
            dtype=np.float32,
        )
        priorities = np.clip(priorities, 1e-8, None)

        # Apply topic boost before normalization
        if topic_boost:
            for i, ep in enumerate(episodes):
                if ep.topic_tags:
                    boost = topic_boost.get(ep.topic_tags[0], 1.0)
                    priorities[i] *= boost

        probs = priorities / priorities.sum()

        n_sample = min(n, len(episodes))
        indices = np.random.choice(len(episodes), size=n_sample, replace=False, p=probs)

        sampled       = [episodes[i] for i in indices]
        sampled_probs = probs[indices]

        N           = total_count
        raw_weights = (1.0 / (N * sampled_probs)) ** beta
        weights     = (raw_weights / raw_weights.max()).astype(np.float32)

        return sampled, weights
```

- [ ] **Step 7: Run all episode store tests**

```
pytest halo_fep/memory/tests/test_episode_store.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add halo_fep/memory/episode_store.py halo_fep/memory/tests/test_episode_store.py
git commit -m "feat(memory): auto-tag topics in add(), get_curiosity_gaps(), topic_boost in get_prioritized()"
```

---

## Task 4: Per-Cluster Wikipedia Iterator

**Files:**
- Modify: `halo_fep/training/topic_bootstrap.py`
- Modify: `halo_fep/training/tests/test_topic_bootstrap.py`

### Background

`iter_wikipedia_token_batches(cfg)` streams all 8 clusters interleaved. Phase 2 needs to stream a single cluster. We add `iter_cluster_token_batches(cfg, cluster_id)` that collects only articles matching `TOPIC_KEYWORDS[cluster_id]`.

- [ ] **Step 1: Write failing test**

Append to `halo_fep/training/tests/test_topic_bootstrap.py`:

```python
def test_iter_cluster_token_batches_returns_correct_shape(monkeypatch):
    """iter_cluster_token_batches yields (n_tokens, d_model) arrays for one cluster."""
    from unittest.mock import patch
    from halo_fep.training.topic_bootstrap import iter_cluster_token_batches
    from halo_fep.config import HaloFEPConfig

    cfg = HaloFEPConfig(n_tokens=4, d_model=8, n_hidden=8)

    # One matching article for cluster 1 (keyword: "algorithm")
    fake_articles = [
        {"text": "This article is about algorithm design and software systems."},
    ]

    import itertools
    with patch("halo_fep.training.topic_bootstrap.load_dataset", return_value=iter(fake_articles)):
        gen = iter_cluster_token_batches(cfg, cluster_id=1, n_articles=1)
        batch = next(gen)

    assert batch.shape == (4, 8)
    assert batch.dtype.kind == "f"


def test_iter_cluster_token_batches_cycles_indefinitely(monkeypatch):
    """Generator cycles past n_articles without raising StopIteration."""
    from unittest.mock import patch
    from halo_fep.training.topic_bootstrap import iter_cluster_token_batches
    from halo_fep.config import HaloFEPConfig
    import itertools

    cfg = HaloFEPConfig(n_tokens=4, d_model=8, n_hidden=8)
    fake_articles = [{"text": "algorithm programming software api"}]

    with patch("halo_fep.training.topic_bootstrap.load_dataset", return_value=iter(fake_articles)):
        gen = iter_cluster_token_batches(cfg, cluster_id=1, n_articles=1)
        batches = [next(gen) for _ in range(5)]   # must not raise

    assert len(batches) == 5
```

- [ ] **Step 2: Run to confirm failure**

```
pytest halo_fep/training/tests/test_topic_bootstrap.py::test_iter_cluster_token_batches_returns_correct_shape -v
```

Expected: FAIL with `ImportError` or `AttributeError`

- [ ] **Step 3: Add `iter_cluster_token_batches` to `topic_bootstrap.py`**

Append at the end of `halo_fep/training/topic_bootstrap.py` (after the `for tok in cycle(all_tokens): yield tok` line):

```python

def iter_cluster_token_batches(
    cfg: HaloFEPConfig,
    cluster_id: int,
    seed: int = 42,
    n_articles: int = 200,
) -> Generator[np.ndarray, None, None]:
    """Yield (n_tokens, d_model) token arrays for a single topic cluster.

    Streams WikiText-103 and retains only articles matching
    TOPIC_KEYWORDS[cluster_id]. Cycles indefinitely after collecting
    n_articles matching articles (or all available if fewer exist).

    Args:
        cfg: Model config (n_tokens, d_model).
        cluster_id: Integer key from TOPIC_KEYWORDS (0–7).
        seed: RNG seed for shuffle.
        n_articles: Target article count before cycling.

    Raises:
        ImportError: If `datasets` is not installed.
        KeyError: If cluster_id is not in TOPIC_KEYWORDS.
    """
    if load_dataset is None:
        raise ImportError(
            "Wikipedia topic bootstrap requires: pip install datasets"
        )

    keywords = TOPIC_KEYWORDS[cluster_id]  # raises KeyError if invalid

    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        use_real_embedder = True
    except Exception:
        st_model = None
        use_real_embedder = False

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-v1",
        split="train",
        streaming=True,
    )

    buffer: list[np.ndarray] = []
    for article in dataset:
        text = article.get("text", "")
        if len(text) < 80:
            continue
        if not any(kw in text.lower() for kw in keywords):
            continue

        if use_real_embedder and st_model is not None:
            chunk_size = max(1, len(text) // cfg.n_tokens)
            chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
            tok = np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32)
            for j, ch in enumerate(chunks[: cfg.n_tokens]):
                tok[j] = _embed_chunk_real(ch, st_model, cfg.d_model)
        else:
            tok = _text_to_tokens(text, cfg.n_tokens, cfg.d_model)

        buffer.append(tok)
        if len(buffer) >= n_articles:
            break

    if not buffer:
        log.warning(
            f"No WikiText-103 articles matched cluster {cluster_id} "
            f"({keywords}). Yielding zero tokens."
        )
        buffer = [np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32)]

    rng = random.Random(seed)
    rng.shuffle(buffer)
    log.info(
        f"Cluster {cluster_id} bootstrap: {len(buffer)} articles buffered."
    )

    for tok in cycle(buffer):
        yield tok
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/training/tests/test_topic_bootstrap.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/topic_bootstrap.py halo_fep/training/tests/test_topic_bootstrap.py
git commit -m "feat(training): iter_cluster_token_batches — single-cluster Wikipedia stream for Phase 2"
```

---

## Task 5: NightlyDreamCycle

**Files:**
- Create: `halo_fep/training/dream_cycle.py`
- Create: `halo_fep/training/tests/test_dream_cycle.py`

- [ ] **Step 1: Write failing tests**

Create `halo_fep/training/tests/test_dream_cycle.py`:

```python
"""Tests for NightlyDreamCycle orchestrator."""
import tempfile
from unittest.mock import MagicMock, patch
import numpy as np
import jax

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.training.dream_cycle import NightlyDreamCycle
from halo_fep.memory.episode_store import EpisodeStore
from halo_fep.memory.learning_journal import LearningJournal


def _make_deps(tmp_path: str):
    cfg     = HaloFEPConfig(
        n_tokens=2,
        nightly_duration_min=1,   # 1-min window for fast tests
        phase1_steps=1,
        phase2_steps_per_cluster=1,
        phase3_steps=1,
    )
    model   = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    store   = EpisodeStore(tmp_path + "/eps")
    journal = LearningJournal(tmp_path + "/jnl", spaced_rep_intervals=cfg.spaced_rep_intervals)
    return cfg, model, store, journal


def test_dream_cycle_run_returns_model_and_log():
    """run() returns (HaloFEPModel, dict) with phase keys."""
    with tempfile.TemporaryDirectory() as d:
        cfg, model, store, journal = _make_deps(d)
        cycle = NightlyDreamCycle(cfg, store, journal)
        key   = jax.random.PRNGKey(1)
        updated_model, log = cycle.run(model, key)
        assert isinstance(updated_model, HaloFEPModel)
        assert "phase1" in log
        assert "phase2" in log
        assert "phase3" in log


def test_dream_cycle_journal_record_called():
    """run() calls journal.record() exactly once."""
    with tempfile.TemporaryDirectory() as d:
        cfg, model, store, journal = _make_deps(d)
        journal.record = MagicMock(wraps=journal.record)
        cycle = NightlyDreamCycle(cfg, store, journal)
        cycle.run(model, jax.random.PRNGKey(2))
        assert journal.record.call_count == 1


def test_dream_cycle_phase1_skips_when_no_due_episodes():
    """Phase 1 logs skipped when no episodes are due."""
    with tempfile.TemporaryDirectory() as d:
        cfg, model, store, journal = _make_deps(d)
        cycle = NightlyDreamCycle(cfg, store, journal)
        _, log = cycle.run(model, jax.random.PRNGKey(3))
        assert log["phase1"].get("status") in ("skipped", "skipped_budget")


def test_dream_cycle_phase3_falls_back_without_yesterday_topics():
    """Phase 3 runs with plain PER when no yesterday journal entry exists."""
    with tempfile.TemporaryDirectory() as d:
        cfg, model, store, journal = _make_deps(d)
        # Add a few episodes so Phase 3 has something to train on
        for i in range(3):
            from halo_fep.memory.schema import Episode
            ep = Episode(
                query=f"algorithm query {i}",
                tokens=np.zeros((2, 256), dtype=np.float32),
                swarm_mu=np.zeros((256, 8), dtype=np.float32),
                free_energy=1.0,
                free_energy_delta=-0.1,
            )
            store.add(ep, query_embed=np.zeros(256, dtype=np.float32))

        cycle = NightlyDreamCycle(cfg, store, journal)
        updated_model, log = cycle.run(model, jax.random.PRNGKey(4))
        assert isinstance(updated_model, HaloFEPModel)
        # Phase 3 should have run (not skipped_budget) given 1-min window
        assert log["phase3"].get("status") != "skipped_budget"


def test_dream_cycle_budget_skip():
    """Phases skip when nightly_duration_min is 0 (deadline already passed)."""
    with tempfile.TemporaryDirectory() as d:
        cfg, model, store, journal = _make_deps(d)
        # Override with effectively zero budget
        import dataclasses
        cfg_zero = dataclasses.replace(cfg, nightly_duration_min=1)
        cycle = NightlyDreamCycle(cfg_zero, store, journal)

        # Patch time.monotonic to simulate expired deadline
        import time as _time
        real_monotonic = _time.monotonic
        call_count = {"n": 0}

        def fake_monotonic():
            call_count["n"] += 1
            # First call (deadline set) returns 0; all subsequent return huge value
            if call_count["n"] == 1:
                return 0.0
            return 1e9   # far past deadline

        with patch("halo_fep.training.dream_cycle.time") as mock_time:
            mock_time.monotonic = fake_monotonic
            _, log = cycle.run(model, jax.random.PRNGKey(5))

        assert log["phase1"].get("status") == "skipped_budget"
        assert log["phase2"].get("status") == "skipped_budget"
        assert log["phase3"].get("status") == "skipped_budget"
```

- [ ] **Step 2: Run to confirm failure**

```
pytest halo_fep/training/tests/test_dream_cycle.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'halo_fep.training.dream_cycle'`

- [ ] **Step 3: Create `halo_fep/training/dream_cycle.py`**

```python
"""Three-phase nightly dream cycle orchestrator.

Replaces the single LoRATrainer nightly call with:
  Phase 1 — Spaced Consolidation  (~20 min)
  Phase 2 — Curiosity Gap-Filling (~20 min)
  Phase 3 — Scaffolded Build      (~15 min)

A time.monotonic() deadline enforces the total nightly budget; phases that
cannot start within 5 minutes of the deadline are skipped and logged.
"""
from __future__ import annotations

import datetime
import itertools
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.memory.episode_store import EpisodeStore
from halo_fep.memory.learning_journal import LearningJournal
from halo_fep.training.lora_trainer import LoRATrainer

log = logging.getLogger(__name__)

_BUDGET_MARGIN_SECS = 300   # 5-minute safety margin before deadline


class NightlyDreamCycle:
    """Orchestrates three-phase nightly learning within a time-bounded window.

    Args:
        cfg:     Model and training config.
        store:   EpisodeStore for reading episodes.
        journal: LearningJournal for forgetting curves and topic history.
    """

    def __init__(
        self,
        cfg: HaloFEPConfig,
        store: EpisodeStore,
        journal: LearningJournal,
    ) -> None:
        self.cfg     = cfg
        self.store   = store
        self.journal = journal

    def run(
        self,
        model: HaloFEPModel,
        key: jnp.ndarray,
    ) -> tuple[HaloFEPModel, dict[str, Any]]:
        """Run the three-phase dream cycle.

        Each phase is independent: if one raises an exception it is logged
        and skipped; the remaining phases still run.

        Returns:
            (updated_model, log_dict) where log_dict has keys
            "phase1", "phase2", "phase3", each a dict with at least "status".
        """
        today    = datetime.date.today().isoformat()
        deadline = time.monotonic() + self.cfg.nightly_duration_min * 60 - _BUDGET_MARGIN_SECS
        log1: dict = {}
        log2: dict = {}
        log3: dict = {}
        trained_topics: list[int] = []

        key, k1 = jax.random.split(key)
        if time.monotonic() < deadline:
            try:
                model, log1 = self._phase1(model, k1, today)
            except Exception as exc:
                log.error(f"Phase 1 failed: {exc}")
                log1 = {"status": "error", "error": str(exc)}
        else:
            log1 = {"status": "skipped_budget"}

        key, k2 = jax.random.split(key)
        if time.monotonic() < deadline:
            try:
                model, log2, trained_topics = self._phase2(model, k2, deadline)
            except Exception as exc:
                log.error(f"Phase 2 failed: {exc}")
                log2 = {"status": "error", "error": str(exc)}
        else:
            log2 = {"status": "skipped_budget"}

        key, k3 = jax.random.split(key)
        if time.monotonic() < deadline:
            try:
                model, log3 = self._phase3(model, k3, today)
            except Exception as exc:
                log.error(f"Phase 3 failed: {exc}")
                log3 = {"status": "error", "error": str(exc)}
        else:
            log3 = {"status": "skipped_budget"}

        self.journal.record(today, trained_topics, log1, log2, log3)
        return model, {"phase1": log1, "phase2": log2, "phase3": log3}

    # ------------------------------------------------------------------
    # Phase 1 — Spaced Consolidation
    # ------------------------------------------------------------------

    def _phase1(
        self,
        model: HaloFEPModel,
        key: jnp.ndarray,
        today: str,
    ) -> tuple[HaloFEPModel, dict]:
        """Replay episodes whose SM-2 interval has elapsed."""
        due = self.journal.get_due_episodes(self.store, today)
        if not due:
            return model, {"status": "skipped", "reason": "no_due_episodes"}

        trainer = LoRATrainer(self.cfg, n_steps=self.cfg.phase1_steps)
        model, info = trainer.run(model, due)
        success = info["loss_after"] <= info["loss_before"]
        self.journal.update_forgetting_state(
            [ep.id for ep in due], today, success=success
        )
        return model, {"status": "done", **info}

    # ------------------------------------------------------------------
    # Phase 2 — Curiosity Gap-Filling
    # ------------------------------------------------------------------

    def _phase2(
        self,
        model: HaloFEPModel,
        key: jnp.ndarray,
        deadline: float,
    ) -> tuple[HaloFEPModel, dict, list[int]]:
        """Train on Wikipedia articles for clusters the agent struggled with."""
        gaps = self.store.get_curiosity_gaps(
            since_days=self.cfg.curiosity_since_days,
            min_encounters=self.cfg.curiosity_min_encounters,
            fe_threshold=self.cfg.curiosity_fe_threshold,
        )
        trained:  list[int] = []
        skipped:  list[int] = []

        for cluster_id in gaps:
            if time.monotonic() >= deadline:
                skipped.append(cluster_id)
                continue
            try:
                model = self._train_cluster(model, cluster_id)
                trained.append(cluster_id)
            except Exception as exc:
                log.error(f"Phase 2 cluster {cluster_id} failed: {exc}")
                skipped.append(cluster_id)

        return model, {"status": "done", "clusters_trained": trained, "clusters_skipped": skipped}, trained

    def _train_cluster(self, model: HaloFEPModel, cluster_id: int) -> HaloFEPModel:
        """Train backbone on one curiosity gap cluster using Wikipedia stream."""
        from halo_fep.training.topic_bootstrap import (
            TOPIC_KEYWORDS,
            iter_cluster_token_batches,
        )
        from halo_fep.memory.schema import Episode

        trainer = LoRATrainer(self.cfg, n_steps=self.cfg.phase2_steps_per_cluster)

        if cluster_id in TOPIC_KEYWORDS:
            gen  = iter_cluster_token_batches(
                self.cfg,
                cluster_id=cluster_id,
                n_articles=self.cfg.phase2_steps_per_cluster,
            )
            episodes = [
                Episode(
                    query=f"phase2_cluster{cluster_id}_step{i}",
                    tokens=tok,
                    swarm_mu=np.zeros(
                        (self.cfg.n_agents, self.cfg.n_hidden), dtype=np.float32
                    ),
                    free_energy=1.0,
                    free_energy_delta=-0.1,
                    topic_tags=[str(cluster_id)],
                )
                for i, tok in enumerate(
                    itertools.islice(gen, self.cfg.phase2_steps_per_cluster)
                )
            ]
            model, _ = trainer.run(model, episodes)
        else:
            # Fallback: hyperbolic pre-training for unlisted clusters
            from halo_fep.training.hyperbolic_pretrain import run_hyperbolic_pretrain
            import jax
            model = run_hyperbolic_pretrain(
                model,
                self.cfg,
                jax.random.PRNGKey(cluster_id),
                n_steps=self.cfg.phase2_steps_per_cluster,
            )

        return model

    # ------------------------------------------------------------------
    # Phase 3 — Scaffolded Build
    # ------------------------------------------------------------------

    def _phase3(
        self,
        model: HaloFEPModel,
        key: jnp.ndarray,
        today: str,
    ) -> tuple[HaloFEPModel, dict]:
        """Train on today's PER episodes, boosting yesterday's topic clusters."""
        midnight_ts = datetime.datetime.combine(
            datetime.date.today(), datetime.time.min
        ).timestamp()

        yesterday_topics = self.journal.get_yesterday_topics(today)
        topic_boost      = (
            {str(c): self.cfg.scaffold_boost for c in yesterday_topics}
            if yesterday_topics
            else None
        )

        episodes, weights = self.store.get_prioritized(
            n=200,
            since_timestamp=midnight_ts,
            alpha=self.cfg.per_alpha,
            beta=self.cfg.per_beta,
            topic_boost=topic_boost,
        )
        if not episodes:
            return model, {"status": "skipped", "reason": "no_episodes_today"}

        trainer = LoRATrainer(self.cfg, n_steps=self.cfg.phase3_steps)
        model, info = trainer.run(model, episodes, per_weights=weights)

        # Enroll today's episodes into the forgetting curve
        self.journal.enroll_episodes([ep.id for ep in episodes], today)

        return model, {"status": "done", **info}
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/training/tests/test_dream_cycle.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/dream_cycle.py halo_fep/training/tests/test_dream_cycle.py
git commit -m "feat(training): NightlyDreamCycle — three-phase spaced/curiosity/scaffolded learning"
```

---

## Task 6: Wire into HeartbeatLoop

**Files:**
- Modify: `halo_fep/main.py`
- Modify: `halo_fep/tests/test_main.py`

- [ ] **Step 1: Write failing test**

Append to `halo_fep/tests/test_main.py`:

```python
def test_heartbeat_uses_dream_cycle_when_provided():
    """HeartbeatLoop._nightly_learning() uses dream_cycle when set."""
    import tempfile
    from unittest.mock import MagicMock
    from halo_fep.main import HeartbeatLoop
    from halo_fep.config import HaloFEPConfig
    from halo_fep.model import HaloFEPModel

    cfg   = HaloFEPConfig(n_tokens=2)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))

    dream_cycle = MagicMock()
    dream_cycle.run.return_value = (model, {"phase1": {}, "phase2": {}, "phase3": {}})

    loop = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=make_mock_perception(),
        memory=make_mock_memory(),
        dream_cycle=dream_cycle,
    )
    loop._nightly_learning()
    dream_cycle.run.assert_called_once()


def test_nightly_window_uses_nightly_duration_min():
    """_is_nightly_window returns True when minute < nightly_duration_min."""
    from halo_fep.main import _is_nightly_window
    import datetime as _dt
    from unittest.mock import patch

    # Simulate 02:45 local time
    fake_now = _dt.datetime(2026, 5, 6, 2, 45)
    with patch("halo_fep.main.datetime") as mock_dt:
        mock_dt.datetime.now.return_value = fake_now
        # default 15 min → minute 45 is outside
        assert not _is_nightly_window(duration_min=15)
        # 60 min → minute 45 is inside
        assert _is_nightly_window(duration_min=60)
```

- [ ] **Step 2: Run to confirm failure**

```
pytest halo_fep/tests/test_main.py::test_heartbeat_uses_dream_cycle_when_provided halo_fep/tests/test_main.py::test_nightly_window_uses_nightly_duration_min -v
```

Expected: FAIL with `TypeError` (unexpected keyword argument `dream_cycle`)

- [ ] **Step 3: Update `_is_nightly_window` to accept duration**

Replace lines 29–32 in `halo_fep/main.py`:

```python
def _is_nightly_window(duration_min: int = 15) -> bool:
    """True between 02:00 and 02:00+duration_min local time."""
    now = datetime.datetime.now()
    return now.hour == 2 and now.minute < duration_min
```

- [ ] **Step 4: Add `dream_cycle` parameter to `HeartbeatLoop.__init__`**

Replace the `__init__` signature and body (lines 38–62) in `halo_fep/main.py`:

```python
    def __init__(
        self,
        cfg: HaloFEPConfig,
        model: HaloFEPModel,
        perception,
        memory,
        llm=None,
        goal_updater=None,
        fep_updater=None,
        lora_trainer=None,
        dream_cycle=None,        # NightlyDreamCycle (preferred over lora_trainer)
        state_compressor=None,
    ) -> None:
        from halo_fep.intellect.state_compressor import StateCompressor
        self.cfg              = cfg
        self.model            = model
        self.carry            = model.init_carry(jax.random.PRNGKey(cfg.seed))
        self.perception       = perception
        self.memory           = memory
        self.llm              = llm
        self.goal_updater     = goal_updater
        self.fep_updater      = fep_updater
        self.lora_trainer     = lora_trainer
        self.dream_cycle      = dream_cycle
        self.state_compressor = state_compressor or StateCompressor(cfg)
        self._prev_fe: float | None = None
        self._nightly_done_date: str | None = None
```

- [ ] **Step 5: Update `tick()` to pass `nightly_duration_min` to window check**

Replace line 121 in `halo_fep/main.py`:

```python
        if _is_nightly_window(self.cfg.nightly_duration_min) and self._nightly_done_date != today:
```

- [ ] **Step 6: Update `_nightly_learning()` to use dream_cycle when available**

Replace lines 149–158 in `halo_fep/main.py`:

```python
    def _nightly_learning(self) -> None:
        if self.dream_cycle is not None:
            log.info("Nightly dream cycle starting.")
            try:
                key = jax.random.PRNGKey(int(time.time()) & 0xFFFFFFFF)
                self.model, info = self.dream_cycle.run(self.model, key)
                log.info(f"Dream cycle done: {info}")
            except Exception as e:
                log.error(f"Dream cycle failed: {e}")
        elif self.lora_trainer is not None:
            log.info("Nightly LoRA learning starting.")
            try:
                episodes = self.memory.get_high_confidence()
                self.model, info = self.lora_trainer.run(self.model, episodes)
                log.info(f"Nightly learning done: {info}")
            except Exception as e:
                log.error(f"Nightly learning failed: {e}")
```

- [ ] **Step 7: Update `main()` to construct and pass `NightlyDreamCycle`**

In the `main()` function, after the existing imports (around line 183), add the dream cycle construction. Replace lines 183–195:

```python
    from halo_fep.perception.pipeline import PerceptionPipeline
    from halo_fep.memory.episode_store import EpisodeStore
    from halo_fep.memory.learning_journal import LearningJournal
    from halo_fep.intellect.llm_bridge import LLMBridge
    from halo_fep.intellect.goal_updater import GoalUpdater
    from halo_fep.intellect.state_compressor import StateCompressor
    from halo_fep.training.fep_updater import FEPUpdater
    from halo_fep.training.lora_trainer import LoRATrainer
    from halo_fep.training.dream_cycle import NightlyDreamCycle

    memory  = EpisodeStore("data/episodes/")
    journal = LearningJournal(
        "data/journal/",
        spaced_rep_intervals=cfg.spaced_rep_intervals,
    )

    loop = HeartbeatLoop(
        cfg              = cfg,
        model            = model,
        perception       = PerceptionPipeline(cfg),
        memory           = memory,
        llm              = LLMBridge(),
        goal_updater     = GoalUpdater(cfg),
        fep_updater      = FEPUpdater(cfg),
        lora_trainer     = LoRATrainer(cfg),   # fallback if dream_cycle absent
        dream_cycle      = NightlyDreamCycle(cfg, memory, journal),
        state_compressor = StateCompressor(cfg),
    )
```

- [ ] **Step 8: Run all tests**

```
pytest halo_fep/ -v --tb=short
```

Expected: all tests pass (no new failures).

- [ ] **Step 9: Commit**

```bash
git add halo_fep/main.py halo_fep/tests/test_main.py
git commit -m "feat(main): wire NightlyDreamCycle into HeartbeatLoop; _is_nightly_window uses nightly_duration_min"
```

---

## Self-Review

**Spec coverage:**
- Config extensions ✓ (Task 1)
- LearningJournal with SM-2, get_due_episodes, get_yesterday_topics, record, enroll_episodes ✓ (Task 2)
- EpisodeStore auto-tagging in add(), get_curiosity_gaps(), topic_boost in get_prioritized() ✓ (Task 3)
- iter_cluster_token_batches for Phase 2 per-cluster Wikipedia stream ✓ (Task 4)
- NightlyDreamCycle with all three phases, budget deadline, exception isolation ✓ (Task 5)
- HeartbeatLoop wiring, _is_nightly_window duration, main() construction ✓ (Task 6)
- All test files ✓

**Type consistency check:**
- `LearningJournal.get_due_episodes(store, today: str)` → used in `_phase1` ✓
- `LearningJournal.update_forgetting_state(ids: list[str], today: str, success: bool)` → used in `_phase1` ✓
- `LearningJournal.enroll_episodes(ids: list[str], today: str)` → used in `_phase3` ✓
- `LearningJournal.get_yesterday_topics(today: str) -> list[int]` → `{str(c): boost for c in topics}` in `_phase3` ✓
- `EpisodeStore.get_curiosity_gaps(...) -> list[int]` → iterated in `_phase2` ✓
- `EpisodeStore.get_prioritized(..., topic_boost: dict[str, float] | None)` → called in `_phase3` with `{str(c): ...}` keys (matching `topic_tags[0]` which stores string cluster IDs) ✓
- `iter_cluster_token_batches(cfg, cluster_id: int, ...)` → called in `_train_cluster` ✓
