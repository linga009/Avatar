# Memory Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add island compression (learned projection + echo) and somatic recall (carry-state similarity retrieval on self_surprise) to Avatar's memory pipeline.

**Architecture:** When Page memory's 32-slot island fills, compress via mean + learned refinement → store summary in SQLite + echo faded trace into fresh island. On self_surprise > 0.5, project carry state → query stored summaries by cosine similarity → inject into carry (physics echo) and PFC prompt (narrative context).

**Tech Stack:** JAX/Equinox, SQLite, numpy

---

### Task 1: Add config parameters

**Files:**
- Modify: `halo3/config.py`

- [ ] **Step 1: Add memory pipeline config fields**

After line 43 (`island_size: int = 32`), add:

```python
    # Memory pipeline (v4.3)
    recall_embed_dim: int = 128
    echo_gate_warmup: int = 100
    recall_similarity_threshold: float = 0.3
```

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/config.py
git commit -m "feat(config): add memory pipeline params for island compression + somatic recall"
```

---

### Task 2: Island compression in PageCurveMemory

**Files:**
- Modify: `halo3/page_memory.py`
- Test: `halo3/tests/test_page_memory.py`

- [ ] **Step 1: Write failing tests**

Create `halo3/tests/test_page_memory.py`:

```python
"""Tests for Page memory island compression."""
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.page_memory import PageCurveMemory


def test_compress_island_shape():
    """compress_island returns (d_model,) summary vector."""
    cfg = Halo3Config()
    mem = PageCurveMemory(cfg)
    island = jax.random.normal(jax.random.PRNGKey(0), (cfg.island_size, cfg.d_model))
    summary = mem.compress_island(island)
    assert summary.shape == (cfg.d_model,), f"Expected ({cfg.d_model},), got {summary.shape}"


def test_compress_island_not_zero():
    """Compressed summary should be nonzero for nonzero input."""
    cfg = Halo3Config()
    mem = PageCurveMemory(cfg)
    island = jax.random.normal(jax.random.PRNGKey(1), (cfg.island_size, cfg.d_model))
    summary = mem.compress_island(island)
    assert jnp.any(summary != 0), "Summary should not be all zeros"


def test_echo_seeds_fresh_island():
    """After apply_echo, island[0] should contain the faded summary."""
    cfg = Halo3Config()
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    summary = jax.random.normal(jax.random.PRNGKey(2), (cfg.d_model,))
    new_state = mem.apply_echo(state, summary, tick=200)
    # island[0] should be nonzero (contains echo), rest should be zero
    assert jnp.any(new_state.island[0] != 0), "Echo should seed island[0]"
    assert jnp.allclose(new_state.island[1:], 0), "Rest of island should be zero"
    assert new_state.island_ptr == jnp.int32(1), "Pointer should be at 1"


def test_echo_gate_warmup():
    """During warmup (tick < 100), echo gate is clamped to [0, 0.1]."""
    cfg = Halo3Config()
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    summary = jnp.ones((cfg.d_model,)) * 100.0
    new_state = mem.apply_echo(state, summary, tick=50)
    # With gate clamped to max 0.1, island[0] norm should be <= 0.1 * ||summary||
    echo_norm = jnp.linalg.norm(new_state.island[0])
    summary_norm = jnp.linalg.norm(summary)
    assert echo_norm <= 0.11 * summary_norm, f"Echo too strong during warmup: {echo_norm} vs {0.1 * summary_norm}"


def test_echo_gate_post_warmup():
    """After warmup, echo gate is clamped to [0, 0.5]."""
    cfg = Halo3Config()
    mem = PageCurveMemory(cfg)
    state = mem.init_state()
    summary = jnp.ones((cfg.d_model,)) * 100.0
    new_state = mem.apply_echo(state, summary, tick=200)
    echo_norm = jnp.linalg.norm(new_state.island[0])
    summary_norm = jnp.linalg.norm(summary)
    assert echo_norm <= 0.51 * summary_norm, f"Echo too strong post-warmup: {echo_norm}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_page_memory.py -v`
Expected: FAIL — `compress_island` and `apply_echo` don't exist yet.

- [ ] **Step 3: Implement island compression**

Replace `halo3/page_memory.py` entirely:

```python
"""PageCurveMemory — ring buffer with island eviction and compression."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class PageMemState(NamedTuple):
    cache: jnp.ndarray
    n_cached: jnp.ndarray
    island: jnp.ndarray
    island_ptr: jnp.ndarray


class PageCurveMemory(eqx.Module):
    max_cache: int = eqx.field(static=True)
    island_size: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    echo_gate_warmup: int = eqx.field(static=True)
    W_refine: jnp.ndarray
    g_echo_raw: jnp.ndarray  # raw value, sigmoid applied at use

    def __init__(self, cfg: Halo3Config) -> None:
        self.max_cache = cfg.max_cache
        self.island_size = cfg.island_size
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.echo_gate_warmup = cfg.echo_gate_warmup
        # Learned refinement: mean pool + linear correction
        self.W_refine = jnp.zeros((cfg.d_model, cfg.d_model)) * 0.01
        # Echo gate: initialized so sigmoid(raw) ≈ 0.01
        self.g_echo_raw = jnp.array(-4.6)  # sigmoid(-4.6) ≈ 0.01

    def init_state(self) -> PageMemState:
        return PageMemState(
            cache=jnp.zeros((self.max_cache, self.d_model)),
            n_cached=jnp.int32(0),
            island=jnp.zeros((self.island_size, self.d_model)),
            island_ptr=jnp.int32(0),
        )

    def __call__(self, x_i, state):
        write_ptr = jnp.int32(state.n_cached % self.max_cache)
        new_cache = state.cache.at[write_ptr].set(x_i)
        is_full = state.n_cached >= self.max_cache
        sq = jnp.sum(new_cache ** 2, axis=-1)
        qr = jnp.sum(new_cache ** 4, axis=-1)
        pr = sq ** 2 / (qr + 1e-12)
        s_gen = sq * pr
        evict_idx = jnp.argmin(s_gen)
        evicted = new_cache[evict_idx]
        iptr = jnp.int32(state.island_ptr % self.island_size)
        new_island = jnp.where(is_full, state.island.at[iptr].set(evicted), state.island)
        new_iptr = jnp.where(is_full, jnp.int32(state.island_ptr + 1), state.island_ptr)
        return PageMemState(cache=new_cache, n_cached=jnp.int32(state.n_cached + 1),
                            island=new_island, island_ptr=new_iptr)

    def compress_island(self, island: jnp.ndarray) -> jnp.ndarray:
        """Compress 32 island vectors into a single summary vector."""
        summary = jnp.mean(island, axis=0)  # (d_model,)
        summary = summary + self.W_refine @ summary  # learned refinement
        return summary

    def is_island_full(self, state: PageMemState) -> bool:
        """Check if island buffer has wrapped (ptr >= island_size)."""
        return int(state.island_ptr) >= self.island_size

    def apply_echo(self, state: PageMemState, summary: jnp.ndarray,
                   tick: int) -> PageMemState:
        """Reset island with faded echo of compressed summary."""
        g = jax.nn.sigmoid(self.g_echo_raw)
        # Clamp gate based on warmup
        max_gate = jnp.where(tick < self.echo_gate_warmup, 0.1, 0.5)
        g = jnp.clip(g, 0.0, max_gate)
        faded = g * summary
        new_island = jnp.zeros_like(state.island)
        new_island = new_island.at[0].set(faded)
        return PageMemState(
            cache=state.cache,
            n_cached=state.n_cached,
            island=new_island,
            island_ptr=jnp.int32(1),
        )
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_page_memory.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/page_memory.py halo3/tests/test_page_memory.py
git commit -m "feat(memory): island compression with learned refinement and echo gate"
```

---

### Task 3: Island summary storage in EpisodeStore

**Files:**
- Modify: `halo3/memory/episode_store.py`
- Test: `halo3/tests/test_episode_store.py`

- [ ] **Step 1: Write failing tests**

Create `halo3/tests/test_episode_store.py`:

```python
"""Tests for island summary storage and somatic recall retrieval."""
import os
import tempfile
import numpy as np
from halo3.memory.episode_store import EpisodeStore


def test_save_island_summary():
    """save_island_summary stores and retrieves correctly."""
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(db_path=os.path.join(d, "test.db"))
        summary = np.random.randn(2048).astype(np.float32)
        carry_emb = np.random.randn(128).astype(np.float32)
        store.save_island_summary(tick=100, summary=summary,
                                  carry_embedding=carry_emb, topic="quantum")
        results = store.get_island_summaries()
        assert len(results) == 1
        assert results[0]["tick"] == 100
        assert results[0]["topic"] == "quantum"
        np.testing.assert_allclose(results[0]["summary"], summary, atol=1e-6)
        np.testing.assert_allclose(results[0]["carry_embedding"], carry_emb, atol=1e-6)


def test_retrieve_similar_island():
    """retrieve_similar_island returns best match by cosine similarity."""
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(db_path=os.path.join(d, "test.db"))
        # Store two summaries with distinct embeddings
        emb_a = np.array([1.0, 0.0, 0.0] + [0.0] * 125, dtype=np.float32)
        emb_b = np.array([0.0, 1.0, 0.0] + [0.0] * 125, dtype=np.float32)
        store.save_island_summary(tick=10, summary=np.zeros(2048, dtype=np.float32),
                                  carry_embedding=emb_a, topic="physics")
        store.save_island_summary(tick=20, summary=np.zeros(2048, dtype=np.float32),
                                  carry_embedding=emb_b, topic="biology")
        # Query close to emb_a
        query = np.array([0.9, 0.1, 0.0] + [0.0] * 125, dtype=np.float32)
        result = store.retrieve_similar_island(query, threshold=0.3)
        assert result is not None
        assert result["tick"] == 10
        assert result["topic"] == "physics"


def test_retrieve_similar_island_below_threshold():
    """Returns None when no island exceeds similarity threshold."""
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(db_path=os.path.join(d, "test.db"))
        emb = np.array([1.0, 0.0, 0.0] + [0.0] * 125, dtype=np.float32)
        store.save_island_summary(tick=10, summary=np.zeros(2048, dtype=np.float32),
                                  carry_embedding=emb, topic="physics")
        # Query orthogonal to stored
        query = np.array([0.0, 0.0, 1.0] + [0.0] * 125, dtype=np.float32)
        result = store.retrieve_similar_island(query, threshold=0.3)
        assert result is None


def test_retrieve_similar_island_empty():
    """Returns None when no summaries stored."""
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(db_path=os.path.join(d, "test.db"))
        query = np.random.randn(128).astype(np.float32)
        result = store.retrieve_similar_island(query, threshold=0.3)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_episode_store.py -v`
Expected: FAIL — `save_island_summary`, `get_island_summaries`, `retrieve_similar_island` don't exist.

- [ ] **Step 3: Implement island summary table and methods**

In `halo3/memory/episode_store.py`, add to `_create_tables()` (after line 61, before `self.conn.commit()`):

```python
        # Island summaries: compressed Page memory epochs
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS island_summaries (
                tick INTEGER PRIMARY KEY,
                summary BLOB,
                carry_embedding BLOB,
                topic TEXT
            )
        """)
```

Add these methods after `get_high_confidence` (after line 176):

```python
    def save_island_summary(self, tick: int, summary: np.ndarray,
                            carry_embedding: np.ndarray, topic: str) -> None:
        """Store a compressed island summary with its carry embedding."""
        self.conn.execute(
            "INSERT OR REPLACE INTO island_summaries (tick, summary, carry_embedding, topic) "
            "VALUES (?, ?, ?, ?)",
            (tick, summary.tobytes(), carry_embedding.tobytes(), topic),
        )
        self.conn.commit()

    def get_island_summaries(self) -> list[dict]:
        """Get all stored island summaries."""
        rows = self.conn.execute(
            "SELECT tick, summary, carry_embedding, topic FROM island_summaries ORDER BY tick"
        ).fetchall()
        results = []
        for r in rows:
            results.append({
                "tick": r[0],
                "summary": np.frombuffer(r[1], dtype=np.float32).copy(),
                "carry_embedding": np.frombuffer(r[2], dtype=np.float32).copy(),
                "topic": r[3],
            })
        return results

    def retrieve_similar_island(self, query_embedding: np.ndarray,
                                threshold: float = 0.3) -> dict | None:
        """Find the most similar island summary by carry embedding cosine similarity."""
        summaries = self.get_island_summaries()
        if not summaries:
            return None
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        best_sim = -1.0
        best = None
        for s in summaries:
            emb = s["carry_embedding"]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(query_norm, emb_norm))
            if sim > best_sim:
                best_sim = sim
                best = s
        if best_sim >= threshold:
            return best
        return None
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_episode_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/memory/episode_store.py halo3/tests/test_episode_store.py
git commit -m "feat(memory): island summary storage and somatic recall retrieval in EpisodeStore"
```

---

### Task 4: Add W_query to Halo3Model and carry embedding projection

**Files:**
- Modify: `halo3/model.py`

- [ ] **Step 1: Add W_query parameter to Halo3Model**

In `halo3/model.py`, add `W_query` field after `page_memory` (line 32):

```python
    W_query: jnp.ndarray  # (d_model, recall_embed_dim) — carry → query embedding
```

In `__init__` (after line 45), add initialization:

```python
        self.W_query = jax.random.normal(keys[7], (cfg.d_model, cfg.recall_embed_dim)) * 0.02
```

- [ ] **Step 2: Add carry_embedding helper method**

After the `init_carry` method (after line 52), add:

```python
    def carry_embedding(self, carry: Halo3Carry) -> jnp.ndarray:
        """Project current carry state to recall embedding space."""
        cache_mean = jnp.mean(carry.page_mem.cache, axis=0)  # (d_model,)
        return cache_mean @ self.W_query  # (recall_embed_dim,)
```

- [ ] **Step 3: Run existing model tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ -v --tb=short -q 2>&1 | tail -5`
Expected: All existing tests still pass (W_query is a new parameter with no breaking changes)

- [ ] **Step 4: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/model.py
git commit -m "feat(model): add W_query for carry state → recall embedding projection"
```

---

### Task 5: Wire island compression and somatic recall through organism.py

**Files:**
- Modify: `halo3/psyche/organism.py`
- Modify: `halo3/psyche/prefrontal.py`

- [ ] **Step 1: Add island compression trigger to organism tick**

In `organism.py`, add a method to handle island compression. Add after the `_experience_log` initialization in `__init__` (after line ~76):

```python
        self._last_island_ptr: int = 0  # track island wrapping
        self._recall_context: str = ""  # last somatic recall for PFC
```

In the `tick()` method, after the self_surprise block (after line ~189), add the somatic recall trigger:

```python
        # --- Somatic recall: retrieve physically similar past on surprise ---
        self._recall_context = ""
        if self_surprise > 0.5 and hasattr(self, '_memory_ref') and self._memory_ref:
            try:
                import numpy as np
                carry_emb = np.array(jax.device_get(
                    jnp.mean(self._carry_cache, axis=0) @ self._W_query
                )) if hasattr(self, '_carry_cache') else None
                if carry_emb is not None:
                    recalled = self._memory_ref.retrieve_similar_island(
                        carry_emb, threshold=self._cfg.recall_similarity_threshold)
                    if recalled:
                        self._recall_context = (
                            f"Body memory: at tick {recalled['tick']}, "
                            f"you were in a similar physical state while exploring "
                            f"'{recalled['topic']}'."
                        )
                        log.info(f"  ⟲ Somatic recall: tick {recalled['tick']} "
                                 f"({recalled['topic']})")
            except Exception as e:
                log.debug(f"Somatic recall failed: {e}")
```

This needs the carry cache and W_query to be available. These will be set by `main.py` each tick (see Task 6).

- [ ] **Step 2: Pass recall_context to PFC generate_query**

Update the `generate_query` call in `_decide_query` (around line ~431):

```python
        pfc_query = self.prefrontal.generate_query(
            current_query, emotion, r_mean, texts,
            self.self_model.strengths,
            consecutive_failures=self._consecutive_zero_results,
            dead_queries=self.self_model.dead_queries,
            qualifier=self.emotions.qualifier,
            mood=self.emotions.mood,
            recall_context=self._recall_context,
        )
```

- [ ] **Step 3: Add recall_context param to prefrontal.py generate_query**

In `prefrontal.py`, add `recall_context: str = ""` param to `generate_query` signature.

In the prompt construction, after the `dead_warning` append (around line ~486), add:

```python
        if recall_context:
            prompt_parts.append(f"\n{recall_context}")
```

Also add `recall_context: str = ""` to `interpret_finding` and include it in the prompt if present.

- [ ] **Step 4: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/organism.py halo3/psyche/prefrontal.py
git commit -m "feat(psyche): somatic recall trigger on self_surprise + PFC context injection"
```

---

### Task 6: Wire island compression events through main.py

**Files:**
- Modify: `halo3/main.py`

- [ ] **Step 1: Add island compression check after each tick**

In `main.py`, after the `halo3_step` call and before the organism tick, check if the island has wrapped. This is where compression, storage, and echo happen.

Find the section where `carry` is updated after `halo3_step` (the carry assignment). After the carry is updated but before `organism.tick()`, add:

```python
        # --- Island compression: when island fills, compress + store + echo ---
        if model.page_memory.is_island_full(carry.page_mem):
            import numpy as np
            # Compress
            summary = model.page_memory.compress_island(carry.page_mem.island)
            # Carry embedding for later somatic recall
            carry_emb = model.carry_embedding(carry)
            carry_emb_np = np.array(jax.device_get(carry_emb))
            summary_np = np.array(jax.device_get(summary))
            # Store in episode SQLite
            memory.save_island_summary(
                tick=tick,
                summary=summary_np,
                carry_embedding=carry_emb_np,
                topic=current_query[:50],
            )
            # Echo: reset island with faded trace
            new_page_mem = model.page_memory.apply_echo(carry.page_mem, summary, tick=tick)
            carry = carry._replace(page_mem=new_page_mem)
            log.info(f"  ◈ Island compressed at tick {tick} — stored + echoed")
```

Also pass carry cache and W_query to organism for somatic recall:

```python
        # Expose carry state for somatic recall
        organism._carry_cache = carry.page_mem.cache
        organism._W_query = model.W_query
        organism._memory_ref = memory
```

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/main.py
git commit -m "feat(main): wire island compression events — compress, store, echo per island cycle"
```

---

### Task 7: Full test suite + integration verification

**Files:**
- All test files

- [ ] **Step 1: Run full test suite**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ tests/ -v --tb=short 2>&1 | tail -30`
Expected: All tests pass (200+ existing + new page_memory + episode_store tests)

- [ ] **Step 2: Fix any failures**

Search for breakage from new `W_query` param or `PageCurveMemory` signature changes:

```bash
cd D:/New_Ai/.worktrees/halo3 && grep -rn "PageCurveMemory\|Halo3Model(" halo3/tests/ tests/ --include="*.py"
```

If any tests construct `Halo3Model` directly, they may need the extra `keys` split for `W_query`. Fix as needed.

- [ ] **Step 3: Commit fixes**

```bash
cd D:/New_Ai/.worktrees/halo3
git add -u
git commit -m "fix: update tests for memory pipeline parameter changes"
```

---

### Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add memory pipeline docs**

In the Architecture section, under Page memory, add:

```
Island compression (v4.3): when 32-slot island fills, compress via mean + learned W_refine,
store summary in SQLite, echo faded trace back via learned g_echo gate (clamped [0,0.1] warmup,
[0,0.5] after). Somatic recall: on self_surprise > 0.5, retrieve past island summaries by
carry-state cosine similarity (W_query projection), inject into PFC prompt + carry echo.
```

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add CLAUDE.md
git commit -m "docs: add memory pipeline (island compression + somatic recall) to CLAUDE.md"
```
