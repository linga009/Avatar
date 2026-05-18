# FineWeb-Edu Integration Design

**Date**: 2026-05-18
**Status**: Approved
**Scope**: Replace web search with FineWeb-Edu Parquet as perception source + add FineWeb batch training phase to dreams (Option C)

---

## Problem

Avatar's perception pipeline depends on live web search (DuckDuckGo + Wikipedia + arXiv). This causes:
- Tick overruns of 160–320s on a 70s target interval (2–5x slowdown)
- Stuck query loops when PFC adapter generates malformed queries (Bug #1)
- Unreliable training signal (search results are noisy and query-dependent)

## Solution

Replace web search with a local FineWeb-Edu Parquet source for perception, and add FineWeb batch training as Phase 4 of the body dream cycle.

---

## Architecture

### Data Flow (Waking — replaces web search)

```
query
  └── ParquetSource.search(query, n=5)
        ├── keyword index lookup (word overlap against all rows)
        ├── fallback: sequential cursor if no keyword match
        └── list[SearchResult]  ← identical interface to web_search()
              └── pipeline.py (unchanged downstream)
```

### Data Flow (Dream — new Phase 4)

```
dream_worker.py
  ├── Phase 1: Replay        (existing, 30 steps)
  ├── Phase 2: Recombine     (existing, 15 steps)
  ├── Phase 3: Imagine       (existing, 10 steps)
  └── Phase 4: FineWeb batch (new, 40 steps)
        ├── sample random row from Parquet
        ├── NativeEmbedder: text → (32, 2048) token tensor
        ├── _fineweb_step (single @eqx.filter_jit outside loop)
        └── CLion update, scale=0.05, NaN guard
```

---

## Files

### New files

**`scripts/download_fineweb.py`** (runs on host)
- Downloads 3 Parquet shards of `HuggingFaceFW/fineweb-edu` `sample-10BT` via `huggingface_hub`
- Target: `data/fineweb/` (~1.2GB total)
- No filtering at download time (filtering happens at load)

**`halo3/perception/parquet_source.py`**
- `ParquetSource` class
- Loads all `*.parquet` files from a directory using `pyarrow.parquet`
- Filters rows: `int_score >= 3` at load time
- Builds inverted keyword index: `word (lowercased, len>=4) → list[int]` (row indices)
- `search(query, n=5) → list[SearchResult]`: word-overlap match, rank by overlap score, return top-n
- Sequential cursor fallback when no keyword match found
- `SearchResult(title=first_20_words, snippet=text[:500], url=url_or_id)`

**`halo3/training/dream_fineweb.py`**
- `fineweb_dream_phase(model, opt_state, opt, carry, parquet_dir, n_steps=40, scale=0.05, seed=0)`
- Loads ParquetSource (reuses same class)
- Single `@eqx.filter_jit` `_fineweb_step` defined once outside loop (prevents XLA OOM)
- Same `_safe_step` NaN guard pattern as `dream_replay.py`
- Uses NativeEmbedder to convert text → token tensors
- Returns `(model, opt_state, info_dict)`

### Modified files

**`halo3/perception/pipeline.py`**
- `__init__`: check for `data/fineweb/*.parquet` at startup
- If found: instantiate `ParquetSource`, use in `perceive()` instead of `web_search`
- If not found: use `web_search` (existing behavior, graceful degradation)
- Log which source is active at startup

**`halo3/training/dream_worker.py`**
- Add `--fineweb-steps` argument (default: 40, 0 = disabled)
- After `dream_replay_physics()`, call `fineweb_dream_phase()` if `--fineweb-steps > 0` and `data/fineweb/` exists

**`halo3/training/dream_replay.py`**
- No changes needed (FineWeb phase is separate)

**`main.py`**
- Pass `--fineweb-steps 40` when spawning dream_worker subprocess

---

## Key Constraints

- **Single JIT outside loop**: `_fineweb_step` must be defined once outside the training loop, with `scale` passed as a `jnp.float32()` argument — not captured by closure. Violating this causes XLA recompilation OOM (same bug fixed in dream_replay.py).
- **opt_state as explicit arg**: pass `opt_state` as explicit parameter to JIT, never closure-captured (same bug fixed in dream_replay.py).
- **NaN guard**: all FineWeb steps wrapped in `_safe_step` equivalent — skip step if loss or weights contain NaN.
- **pyarrow already installed**: in Docker image, no Dockerfile changes needed.
- **Volume already mounted**: `./data:/app/data` covers `data/fineweb/` automatically.

---

## Performance

| Metric | Before | After |
|---|---|---|
| Tick duration | 160–320s | ~70s (no HTTP) |
| Rows/day (waking) | ~50 (unreliable web) | ~300 (fast local) |
| Rows/day (dream) | 0 | ~480 (40 steps × 12 dreams) |
| Dream duration | ~60s | ~105s (+45s FineWeb phase) |
| Stuck query loops | Frequent | Eliminated (no HTTP) |

---

## Download Instructions

Run on host (not in Docker):
```bash
cd D:/New_Ai/.worktrees/halo3
python scripts/download_fineweb.py
```

The organism auto-detects the data at next container startup. A container restart is required since `PerceptionPipeline` is initialized once at process start.

---

## Out of Scope

- FAISS semantic search over Parquet (overkill for this architecture)
- Downloading full sample-10BT (1.2GB from 3 shards is sufficient)
- Stochastic row chunking (one row = one gradient step, no windowing)
- Web search as hybrid fallback (pure replacement, Option A)
