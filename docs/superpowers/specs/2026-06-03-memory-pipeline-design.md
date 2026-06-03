# Memory Pipeline Improvements: Island Compression + Somatic Recall

**Date:** 2026-06-03
**Status:** Design approved

## Problem

Avatar's Page memory evicts entries to a 32-slot island buffer. When the island fills, old entries are silently overwritten — all compressed history is lost. And when Avatar encounters surprising situations (high ΔFE), it has no mechanism to recall past experiences that felt physically similar.

## Design Principle

Compress, store, echo. The body remembers by resonance, not by keyword. Memories are both stored (retrievable) and felt (echoed into carry state). The PFC translates what the body recalls.

## Part 1: Page Memory Island Compression

### Current Flow

```
Cache full → evict lowest participation-ratio → write to island[32]
                                                  ↓ (when island full)
                                                 overwritten, lost
```

### New Flow

```
Cache full → evict to island → island full → compress → store + echo
                                                          ↓         ↓
                                              Episode SQLite    Carry state
                                              (retrievable)    (felt echo)
```

### Compression

A learned linear projection contracts the 32 island vectors into a single summary vector:

```python
W_compress: jnp.ndarray  # shape (32 * d_model, d_model) — but see implementation note
```

Implementation: reshape island `(32, d_model)` → flatten `(32 * d_model,)` → linear → `(d_model,)`. This is too large for d_model=2048 (32*2048*2048 = 134M params). Instead, use mean pooling + a small learned refinement:

```python
summary = jnp.mean(island, axis=0)  # (d_model,)
summary = summary + W_refine @ summary  # W_refine: (d_model, d_model) — 4.2M params
```

This is ~4.2M params, fits in VRAM, and trained alongside the body during per-tick learning. The mean pooling preserves the bulk of the information; the learned refinement captures correlations.

### Store

The summary vector is serialized and saved to the episode store:

```python
episode_store.save_island_summary(
    tick=current_tick,
    summary=summary,           # (d_model,) = (2048,)
    carry_embedding=carry_emb, # (recall_embed_dim,) = (128,) for retrieval
    topic=current_topic,
)
```

New table `island_summaries` in SQLite:
- `tick` INTEGER PRIMARY KEY
- `summary` BLOB (numpy serialized, ~8KB per entry at float32)
- `carry_embedding` BLOB (512 bytes at float32, dim=128)
- `topic` TEXT

At 128 tokens/tick and 128 cache entries, an island fills every ~128 ticks. Over 10,000 ticks that's ~78 summaries, ~625KB total. Negligible storage.

### Echo

A learned gate blends a faded version of the summary back into the island buffer:

```python
g_echo: float  # scalar, sigmoid-initialized to ~0.01
```

After compression and storage:

```python
# Reset island with faded echo of compressed summary
faded = g_echo * summary  # (d_model,)
new_island = jnp.zeros_like(island)
new_island = new_island.at[0].set(faded)  # seed position 0 with echo
island_ptr = 1  # next eviction writes to position 1
```

The echo seeds the fresh island with a compressed trace of the past. The gate is learned — Avatar discovers how much past to let through.

**Safety:** `g_echo` is clamped to [0, 0.1] during the first 100 ticks (warmup). After warmup, clamped to [0, 0.5] — the echo can never dominate.

## Part 2: ΔFE-Triggered Somatic Recall

### Trigger

`self_surprise > 0.5` — already computed in `organism.py` via the introspective monitor (z-score of tau derivative exceeding 2σ). This is Avatar's existing "something unexpected happened" signal.

### Query Embedding

When island summaries are stored (Part 1), also store a projection of the current carry state:

```python
W_query: jnp.ndarray  # shape (d_model, recall_embed_dim) — (2048, 128)
carry_emb = W_query.T @ current_carry_vector  # (128,)
```

`current_carry_vector` is the mean of the active Page memory cache: `jnp.mean(carry.page_mem.cache, axis=0)`.

~262K params for W_query. Trained alongside the body.

### Retrieval

When self_surprise > 0.5:

1. Project current carry state through W_query → query_emb (128,)
2. Load all stored carry_embeddings from SQLite
3. Cosine similarity: `sim = query_emb · stored_emb / (|query| · |stored|)`
4. Retrieve top-1 match (if similarity > 0.3 threshold)

### Injection (Both Paths)

**Carry echo:** Retrieved summary vector blended into Page memory island slot 0:

```python
g_recall: float  # separate learned gate, same constraints as g_echo
island = island.at[0].set(island[0] + g_recall * retrieved_summary)
```

**PFC context:** The retrieved episode's tick and topic are added to the PFC prompt context in `organism.py`:

```python
if recalled_episode:
    recall_context = (
        f"Body memory: at tick {recalled_episode['tick']}, "
        f"you were in a similar physical state while exploring "
        f"'{recalled_episode['topic']}'."
    )
```

This flows through the existing body-voice pipeline — the PFC translates what the body recalls. Added to `generate_query` and `interpret_finding` prompts alongside qualifier and mood.

## Config Additions

```python
# Memory pipeline (v4.3)
recall_embed_dim: int = 128
echo_gate_warmup: int = 100
recall_similarity_threshold: float = 0.3
```

## Files Changed

| File | Change |
|---|---|
| `halo3/page_memory.py` | Add `W_refine`, `g_echo`, compression + echo logic, `compress_island()` method |
| `halo3/model.py` | Add `W_query`, carry embedding computation, expose island compression events |
| `halo3/memory/episode_store.py` | Add `island_summaries` table, `save_island_summary()`, `retrieve_similar()` |
| `halo3/psyche/organism.py` | Trigger recall on self_surprise, inject into PFC prompt + carry |
| `halo3/psyche/prefrontal.py` | Accept optional `recall_context` param in generate_query/interpret_finding |
| `halo3/config.py` | `recall_embed_dim`, `echo_gate_warmup`, `recall_similarity_threshold` |

## What This Does NOT Change

- COP engine, Kuramoto physics, emotions, body-voice pipeline — untouched
- Dream cycle phases — untouched
- Chat system prompt — untouched (recall context goes through PFC, not chat)
- Knowledge graph — untouched (complementary: graph = semantic structure, recall = physics resonance)
- Existing Page memory eviction logic (participation ratio) — untouched
- XLA compilation graph — `W_refine`, `W_query`, `g_echo`, `g_recall` are static shapes

## Checkpoint Impact

New model parameters:
- `W_refine`: 2048 × 2048 = 4.2M params (~16.8MB float32)
- `W_query`: 2048 × 128 = 262K params (~1MB)
- `g_echo`: 1 scalar
- `g_recall`: 1 scalar

Total: ~4.5M new params, ~18MB checkpoint growth. Main checkpoint goes from 426MB → ~444MB. Fits in VRAM — forward pass adds negligible compute (one matmul per island compression, ~every 128 ticks).

**Backward compatibility:** Old checkpoints will be missing these params. `model.py` init should initialize them from scratch (Xavier for W_refine/W_query, 0.01 for gates). Existing carry state is unaffected.

## Tests

- `test_island_compression`: verify 32 vectors → 1 summary, correct shape (d_model,)
- `test_echo_gate_warmup`: verify clamped to [0, 0.1] during first 100 ticks
- `test_echo_gate_post_warmup`: verify clamped to [0, 0.5] after warmup
- `test_echo_seeds_fresh_island`: verify island[0] contains faded summary after compression
- `test_store_retrieve_roundtrip`: save island summary to SQLite, retrieve by embedding similarity
- `test_recall_trigger`: verify fires on self_surprise > 0.5, returns None otherwise
- `test_recall_similarity_threshold`: verify no retrieval when all similarities < 0.3
- `test_carry_echo_bounded`: verify carry state norm doesn't grow unboundedly after repeated echoes
- `test_pfc_recall_context`: verify recall context string appears in PFC prompt when episode recalled

## Growth Path

After running for 1000+ ticks with this system:
- Avatar will have ~8 island summaries stored, each representing ~128 ticks of compressed experience
- When surprised, it will recall physically similar past states — not topically similar ones
- The learned gates will reveal how much Avatar values its own history
- If g_echo stays near 0, the body doesn't need the past. If it grows toward 0.5, the past is load-bearing.
- Validates the "metabolic pump" metaphor for the ALIFE paper: Avatar processes, compresses, stores, and recalls through physics — not through attention over long sequences
