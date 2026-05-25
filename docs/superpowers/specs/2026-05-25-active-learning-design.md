# Free Energy-Guided Active Learning — Design Spec

**Date:** 2026-05-25
**Status:** Approved
**Scope:** Replace sequential FineWeb curriculum with PFC-directed, FE-scored active learning

## Problem

Avatar's learning from FineWeb-Edu is passive and sequential:
- **Dream Phase 4:** A cursor walks through parquet rows in order. Avatar trains on whatever comes next, regardless of relevance or readiness.
- **Waking ticks:** ParquetSource loads 50K rows into RAM at startup, keyword-matches against PFC queries. PFC generates queries blind — no awareness of what's available.
- **50K row cap:** Only ~7% of the 726K available rows are loaded. The rest is invisible.
- **No self-direction:** Avatar cannot decide what it needs to learn. Learning is input-driven, not need-driven.

## Solution

Apply the Free Energy Principle to curriculum selection. Avatar's own prediction error determines what's worth learning. The body's physics IS the curriculum selector.

**Core insight:** A text with medium prediction error is in Avatar's zone of proximal development — complex enough to learn from, simple enough to integrate. Texts with very low FE (already mastered) and very high FE (incomprehensible noise) are both uninformative.

## Architecture

```
TopicIndex (persisted, one-time build)
    726K+ rows --> ~300-500 topic clusters
    Per cluster: keywords, count, file_positions[]
    Streams from parquet -- never all in RAM
            |
            v
ActiveSampler
    1. BS valuation ranks topic clusters
    2. Pull ~50 candidates from top-valued clusters (stream from disk)
    3. Forward-only pass --> measure FE per text (~5ms each, no gradients)
    4. Select ~10 with medium FE (zone of proximal development)
            |
      +-----+------+
      |            |
  Dream Ph4    Waking Ticks
  (batch)      (per-tick)
```

## Component 1: TopicIndex

**File:** `halo3/perception/topic_index.py`

### Build Process

Runs once per corpus change. Persisted to `data/fineweb/topic_index.json`.

1. Stream every row group from all parquet files (never load full corpus)
2. For each text, extract top-5 keywords:
   - Tokenize to lowercase words (length >= 4)
   - Score by TF-IDF: `tf(word, doc) * log(N / df(word))`
   - IDF computed in first pass (count documents containing each word)
   - Keep top-5 scoring words per text
3. Cluster into topic buckets by keyword overlap:
   - First text creates bucket with its keywords
   - Subsequent texts: if 2+ keywords match an existing bucket, add to it
   - Otherwise create new bucket
   - Merge buckets sharing 3+ keywords (single pass after all texts assigned)
   - Target: 300-500 buckets (tune overlap threshold if needed)
4. Per bucket, store:
   ```json
   {
     "topic_id": 42,
     "keywords": ["quantum", "entanglement", "photon", "coherence", "bell"],
     "row_count": 1847,
     "sample_titles": ["Quantum entanglement explained...", ...],
     "file_positions": [
       {"file": 0, "row_group": 12, "row": 345},
       {"file": 0, "row_group": 12, "row": 891},
       ...
     ]
   }
   ```

### Query Interface

```python
class TopicIndex:
    def __init__(self, index_path: str, parquet_dir: str):
        """Load persisted index. Does NOT load parquet data."""

    @staticmethod
    def build(parquet_dir: str, output_path: str) -> "TopicIndex":
        """Scan corpus and build topic index. Streams row groups."""

    def get_topics(self) -> list[TopicBucket]:
        """Return all topic buckets (lightweight — no text data)."""

    def match_topic(self, query: str) -> list[TopicBucket]:
        """Find topics matching query keywords. For waking tick use."""

    def sample_from_topic(self, topic_id: int, n: int) -> list[str]:
        """Stream n texts from disk for a given topic. Opens parquet, reads specific rows, closes."""

    def sample_from_topics(self, topic_ids: list[int], n_per_topic: int) -> list[str]:
        """Stream texts from multiple topics. For active sampling."""
```

### Constraints

- No new dependencies (numpy + pyarrow only)
- RAM during build: ~50MB (IDF dict + keyword buffers). Texts not held.
- RAM at runtime: ~5-10MB (index JSON loaded, texts streamed on demand)
- Index rebuild: ~2-5 minutes for 726K rows, scales linearly
- Parquet files opened per query, not held open

### Scaling to Full FineWeb

When additional shards are downloaded to `data/fineweb/`:
- Run `TopicIndex.build()` again (or incremental append in future version)
- Index file grows linearly (~15 bytes per file_position pointer)
- 10M rows = ~50MB index, 100M rows = ~500MB index (may need binary format at that scale)

## Component 2: ActiveSampler

**File:** `halo3/training/active_sampler.py`

### Algorithm

```python
def select_curriculum(
    model,
    carry,
    topic_index: TopicIndex,
    volatility_surface: VolatilitySurface,
    competence: dict[str, float],
    embedder: NativeEmbedder,
    n_candidates: int = 50,
    n_train: int = 10,
    key: jax.random.PRNGKey,
) -> list[str]:
    """Select the most informative texts for training.

    Uses BS valuation to pick topics, then FE to score candidates.
    Returns n_train texts in the zone of proximal development.
    """
    # Step 1: Rank ALL topics by Black-Scholes option value
    topics = topic_index.get_topics()
    topic_values = []
    for topic in topics:
        # Use first keyword as topic name for BS lookup
        # BS already handles unknown topics (default IV = 1.0)
        primary_keyword = topic.keywords[0]
        value = volatility_surface.value(primary_keyword)
        topic_values.append((topic, value))

    # Sort by value descending, take top 20 topics
    topic_values.sort(key=lambda x: -x[1])
    top_topics = topic_values[:20]

    # Step 2: Sample candidates from top-valued topics
    # Distribute candidates across topics weighted by value
    candidates_per_topic = max(1, n_candidates // len(top_topics))
    candidate_texts = []
    for topic, _ in top_topics:
        texts = topic_index.sample_from_topic(
            topic.topic_id,
            n=candidates_per_topic,
        )
        candidate_texts.extend(texts)
        if len(candidate_texts) >= n_candidates:
            break
    candidate_texts = candidate_texts[:n_candidates]

    # Step 3: Forward-only pass to measure prediction error
    scored = []
    for text in candidate_texts:
        tokens = embedder.texts_to_tokens([text], n_tokens=model.cfg.n_tokens)
        # Forward only — no gradients, no backward pass
        fe = _forward_only_fe(model, carry, tokens, key)
        scored.append((text, float(fe)))

    # Step 4: Zone of proximal development
    # Sort by FE, take middle 60% (skip bottom 20% and top 20%)
    scored.sort(key=lambda x: x[1])
    lo = len(scored) // 5
    hi = 4 * len(scored) // 5
    zone = scored[lo:hi]

    # Return top n_train from the zone (highest FE within the zone)
    # Rationale: within the learnable zone, prefer the most challenging
    zone.sort(key=lambda x: -x[1])
    return [text for text, _ in zone[:n_train]]
```

### Forward-Only FE Measurement

```python
@eqx.filter_jit
def _forward_only_fe(model, carry, tokens, key):
    """Forward pass only — measures prediction error without gradients.

    Returns scalar free energy (prediction error).
    ~5ms per text on GPU. No gradient storage.
    """
    loss, _ = halo3_loss(model, carry, tokens, key)
    return loss
```

This is already JIT-compiled in the existing codebase. We just call `halo3_loss` without `filter_value_and_grad`. No new compilation needed — same function, inference mode.

### Cost Budget

| Operation | Time | RAM | VRAM |
|---|---|---|---|
| BS valuation (300 topics) | <1ms | negligible | 0 |
| Stream 50 texts from disk | ~200ms | ~100KB peak | 0 |
| Embed 50 texts | ~500ms | ~50MB (embedder) | 0 (CPU) |
| Forward pass 50 texts | ~250ms | negligible | ~3.5GB (model already loaded) |
| **Total** | **~1s** | **~50MB peak** | **0 extra** |

## Component 3: Dream Phase 4 Replacement

**Files modified:** `halo3/training/dream_fineweb.py`, `halo3/training/dream_fineweb_worker.py`

### Current Flow (replaced)

```
cursor reads next 10 texts sequentially --> embed --> CLion train
```

### New Flow

```
ActiveSampler selects 10 texts (FE-guided) --> embed --> CLion train
```

The `dream_fineweb_worker.py` subprocess changes:
1. Load model from checkpoint (same as before)
2. Load topic_index from `data/fineweb/topic_index.json`
3. Load BS valuation state from `data/bs_state.json` (or reconstruct from organism state)
4. Call `select_curriculum()` to pick 10 texts
5. Train with CLion optimizer (same `_safe_step`, same JIT pattern)
6. Save updated model

**Removed:** `_load_cursor()`, `_save_cursor()`, `_sample_texts_cursor()`. The sequential cursor is gone.

**Kept:** CLion optimizer, `_safe_step`, NaN guards, subprocess isolation, `jax.checkpoint` for VRAM.

### BS State for Subprocess

The dream subprocess needs BS valuation data. Two options:
- **Option A:** Save `volatility_surface` state to JSON before dream, load in subprocess
- **Option B:** Pass competence map + topic stats as CLI args, reconstruct BS values in subprocess

**Decision:** Option A — save to `data/dream_training/bs_state.json` before spawning subprocess. Contains: `{topic: {S, sigma, last_r_values}}`. Small file (~10KB), already JSON-serializable.

## Component 4: Waking Tick Integration

**Files modified:** `halo3/perception/pipeline.py`

### Current Flow (replaced)

```
PFC query --> ParquetSource keyword search in 50K loaded rows --> top 5 results
```

### New Flow

```
PFC query --> TopicIndex.match_topic(query) --> stream 10 candidates from disk
          --> forward-only FE score --> pick best 5 --> perceive
```

### Changes to PerceptionPipeline

```python
class PerceptionPipeline:
    def __init__(self, d_model, n_tokens, vocab_size=8000):
        # Replace ParquetSource with TopicIndex
        self._topic_index = None
        index_path = "data/fineweb/topic_index.json"
        if os.path.exists(index_path):
            from halo3.perception.topic_index import TopicIndex
            self._topic_index = TopicIndex(index_path, "data/fineweb")
            log.info("Perception: using TopicIndex (FE-guided active learning)")

    def perceive(self, query, max_results=5, model=None, carry=None, key=None):
        if self._topic_index is not None:
            # Find matching topics
            topics = self._topic_index.match_topic(query)
            if topics:
                # Stream candidates from matching topics
                candidates = self._topic_index.sample_from_topics(
                    [t.topic_id for t in topics[:5]], n_per_topic=3
                )
                # If model available, FE-score candidates
                if model is not None and carry is not None:
                    candidates = self._fe_rank(candidates, model, carry, key, max_results)
                else:
                    candidates = candidates[:max_results]
                # Embed and return
                texts = candidates
                tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
                return tokens, texts
        # Fallback: web search
        ...
```

**API change:** `perceive()` gains optional `model`, `carry`, `key` params for FE scoring. When not provided (e.g., first tick before model loads), falls back to keyword matching within the topic index.

### Changes to main.py

Pass model and carry to perceive() call:

```python
tokens, texts = pipeline.perceive(
    query, model=model, carry=carry, key=subkey
)
```

Minimal change — adds 3 kwargs to an existing call.

## Component 5: Dataset Expansion

**Not blocking the above.** Can be done incrementally at any time.

### Download Script

```bash
# Download additional FineWeb-Edu shards
python3 -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT',
                  split='train', cache_dir='data/fineweb/.cache')
"
```

Or direct parquet download for specific shards:
```bash
# Each shard is ~2GB, 726K rows
huggingface-cli download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --include "sample/10BT/*.parquet" \
    --local-dir data/fineweb/
```

After downloading, rebuild topic index:
```python
TopicIndex.build("data/fineweb", "data/fineweb/topic_index.json")
```

### Storage Budget

| Corpus Size | Disk | Index Size | Build Time |
|---|---|---|---|
| 726K rows (current) | 2.1 GB | ~5 MB | ~3 min |
| 5M rows | ~15 GB | ~35 MB | ~20 min |
| 50M rows | ~150 GB | ~350 MB | ~3 hours |
| Full FineWeb-Edu | ~12 TB | needs binary format | needs incremental build |

**Recommendation:** Start with current 726K. Download 2-3 more shards (~5M rows, ~15GB) once active learning is proven. Full FineWeb requires a different storage strategy (external SSD or streaming from HuggingFace).

## File Deleted

| File | Reason |
|---|---|
| `halo3/perception/parquet_source.py` | Replaced by TopicIndex + ActiveSampler. All functionality superseded. |

## Files Created

| File | Purpose |
|---|---|
| `halo3/perception/topic_index.py` | Topic clustering, index build, streaming text retrieval |
| `halo3/training/active_sampler.py` | FE-guided curriculum selection |

## Files Modified

| File | Change |
|---|---|
| `halo3/training/dream_fineweb.py` | Replace cursor reads with `select_curriculum()` |
| `halo3/training/dream_fineweb_worker.py` | Load topic_index, pass to `fineweb_dream_phase()` |
| `halo3/perception/pipeline.py` | Replace ParquetSource with TopicIndex + FE scoring |
| `halo3/main.py` | Pass model/carry to `perceive()`, save BS state before dream |

## Testing Strategy

1. **TopicIndex build:** Verify 726K rows produce 300-500 buckets, all rows assigned, pointers valid
2. **TopicIndex streaming:** Verify `sample_from_topic()` returns correct texts from disk
3. **ActiveSampler:** Mock model with known FE values, verify zone-of-proximal-development filtering
4. **Integration:** Full dream cycle completes with active learning, no OOM, loss decreases
5. **Waking:** Perceive returns FE-ranked results, not keyword-only

## Migration

1. Build topic index (one-time): `python3 -m halo3.perception.topic_index --build`
2. Deploy new code (docker compose build)
3. Old cursor file (`fineweb_cursor.json`) becomes unused — can delete
4. ParquetSource no longer loaded at startup — 500MB RAM freed immediately

## Risks

- **Topic clustering quality:** Simple keyword overlap may produce too-broad or too-narrow clusters. Mitigation: tune overlap threshold, add merge/split heuristics. Can always rebuild.
- **BS valuation cold start:** New topics from expanded corpus have no BS history. Mitigation: default IV=1.0 already handles this (optimistic exploration prior).
- **Forward-only scoring adds ~1s per dream:** 50 forward passes. Acceptable given dream takes 2+ minutes total.
- **Parquet I/O per tick:** Streaming 10-15 texts per tick from disk instead of RAM lookup. SSD latency ~0.1ms per read. Acceptable.
