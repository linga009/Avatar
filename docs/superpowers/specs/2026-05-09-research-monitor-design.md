# HoloBiont Eye — Autonomous Research Monitor Design Spec

> Connect HoloBiont 3.0 physics engine to real-world web perception.
> Kuramoto synchronization detects emerging patterns in research streams.

---

## Goal

A persistent 24/7 process that fetches web content on configurable topics,
processes it through the trained HoloBiont 3.0 backbone + Hamiltonian ODE +
Bohmian Kuramoto swarm, and surfaces findings when oscillators synchronize.

## Architecture

```
topics.yaml → Web Fetch → Embed → (n_tokens, d_model) tensor
                                        │
                                        ▼
                              halo3_step (existing)
                              backbone + Hamiltonian + Kuramoto
                                        │
                                        ▼
                              Interpreter: r → finding/explore
                                        │
                                        ▼
                              Episode Store (SQLite + FAISS)
                                        │
                              Nightly Dreaming (existing trainer)
```

No changes to backbone, Hamiltonian, Kuramoto, or training code.

---

## 1. Perception Pipeline

### embedder.py

Uses sentence-transformers `all-MiniLM-L6-v2` (384-dim output). A learned
linear projection (384 → d_model) converts to model space. The projection
is saved/loaded alongside the model checkpoint.

For n_tokens=32 with 5 search results: each result produces ~6 token
embeddings (title chunks + snippet chunks), padded/truncated to exactly 32.

### web_fetch.py

Uses `duckduckgo_search` library (no API key needed). Takes a query string,
returns list of `{title, snippet, url}` dicts. Timeout 10s per query.
Returns empty list on failure (never crashes the heartbeat).

### pipeline.py

Orchestrates: read current topic → web_fetch → chunk text → embed →
stack into (n_tokens, d_model) tensor. Also provides `embed_query(text)`
for FAISS indexing (returns 384-dim vector).

---

## 2. Output Interpreter

### interpreter.py

Reads the Kuramoto state after each tick:

- `order_parameter(theta)` → r ∈ [0,1] per hidden dimension
- `mean(r)` → scalar confidence
- Phase velocity → which topic dimensions are active

Decision logic:
- r_mean > 0.6 → PATTERN DETECTED: log finding, keep searching this topic
- r_mean < 0.4 → NOISE: rotate to next topic from config
- 0.4 ≤ r_mean ≤ 0.6 → UNCERTAIN: continue current topic

Next query generation:
- In exploit mode: refine current query with top phase velocity keywords
- In explore mode: pick next topic from seed list (round-robin)

---

## 3. Episode Store

### schema.py

```python
@dataclass
class Episode:
    query: str
    tokens: np.ndarray           # (n_tokens, d_model) float32
    order_param: float           # mean r
    mode: str                    # "explore" / "exploit"
    finding: str | None          # description if pattern detected
    timestamp: str               # ISO format
    query_embed: np.ndarray      # (384,) for FAISS
```

### episode_store.py

- SQLite for structured storage (query, timestamp, r, finding)
- FAISS IndexFlatIP for semantic similarity search on query embeddings
- `add(episode)` — insert + index
- `retrieve(query_embed, k=5)` — find similar past episodes
- `get_high_confidence(threshold=0.6)` — episodes with r > threshold for dreaming

---

## 4. Main Heartbeat

### main.py

```
while not shutdown:
    1. Pick topic (from interpreter state)
    2. Perception: fetch + embed → tokens
    3. halo3_step(model, carry, tokens, key) → new_carry, outputs
    4. Interpret: r, mode, finding, next_query
    5. Store episode
    6. Log to terminal
    7. Check nightly window → dream if due
    8. Sleep(tick_interval - elapsed)
```

Signal handling: SIGINT/SIGTERM for graceful shutdown.
Loads checkpoint from `data/checkpoints/halo3.eqx`.

---

## 5. Topic Config

### topics.yaml

```yaml
seed_topics:
  - "artificial intelligence research breakthroughs 2026"
  - "quantum computing error correction advances"
  - "tensor networks machine learning applications"
  - "AdS CFT holography neural networks"
  - "autonomous AI agents research"
  - "physics-informed neural networks"
max_results_per_query: 5
tick_interval: 60
r_exploit_threshold: 0.6
r_explore_threshold: 0.4
```

---

## 6. Docker Update

Add to Dockerfile pip install layer:
```
sentence-transformers duckduckgo-search pyyaml faiss-cpu
```

Add new CMD for monitor mode:
```dockerfile
CMD ["python3", "halo3/main.py"]
```

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `halo3/perception/__init__.py` | Create | Package |
| `halo3/perception/embedder.py` | Create | Text → d_model tensor |
| `halo3/perception/web_fetch.py` | Create | DuckDuckGo search |
| `halo3/perception/pipeline.py` | Create | Orchestrate fetch+embed |
| `halo3/perception/interpreter.py` | Create | r → findings, next query |
| `halo3/memory/__init__.py` | Create | Package |
| `halo3/memory/schema.py` | Create | Episode dataclass |
| `halo3/memory/episode_store.py` | Create | SQLite + FAISS |
| `halo3/main.py` | Create | Heartbeat loop |
| `halo3/topics.yaml` | Create | Seed topics config |
| `Dockerfile` | Modify | Add deps |
| `halo3/tests/test_perception.py` | Create | Embedder + interpreter tests |
| `halo3/tests/test_memory.py` | Create | Episode store tests |
