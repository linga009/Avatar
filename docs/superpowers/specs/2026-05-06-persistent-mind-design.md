# Persistent Mind Design

**Date:** 2026-05-06
**Goal:** A local autonomous agent that runs 24/7 on a 6GB VRAM machine, browses the web driven by free energy minimization, accumulates episodic memory, and self-improves at four timescales simultaneously.

---

## 1. Problem Statement

The Trinity master plan cannot run on 6GB VRAM — Hermes-8B alone consumes ~4.5GB, leaving insufficient headroom for HALO+FEP+JAX runtime. More fundamentally, the Trinity plan is optimized for research elegance rather than practical autonomy. The self-modification engine (LLM writes code → pytest → restart) is unsafe and unlikely to produce useful improvements. The web explorer is a secondary trigger, not the primary perception mechanism.

This design replaces the Trinity plan with an architecture purpose-built for a persistent local mind on consumer hardware.

---

## 2. Core Insight

HALO+FEP is the always-on subconscious. The LLM is a tool it wakes up, not the mind itself. Free energy minimization IS the drive that motivates web exploration — high surprise on a topic means search for more information. This is mathematically grounded in the FEP framework rather than bolted on as a trigger condition.

---

## 3. Architecture

### 3.1 Overview

```
                 HEARTBEAT LOOP (main.py) — runs forever
                          |
                          | every ~60s
          +---------------v------------------+
          |        SUBCONSCIOUS TICK         |
          |  FEP drives -> search query      |
          |  Web -> embed -> (N_tok, d_model)|
          |  halo_fep_step() -> new beliefs  |
          |  Store episode in FAISS (CPU)    |
          +---------------+------------------+
                          |
                          | if free_energy > wake_threshold
          +---------------v------------------+
          |           WAKE CYCLE             |
          |  Compress HALO state -> prompt   |
          |  + retrieve top-5 memories       |
          |  Phi-3.5-mini reasons            |
          |  -> new goal / refined query     |
          |  Update FEP C matrix             |
          |  Unload LLM                      |
          +---------------+------------------+
                          |
                          | nightly (~2am)
          +---------------v------------------+
          |         LEARNING CYCLE           |
          |  LoRA fine-tune HALO SSM layers  |
          |  on high-confidence episodes     |
          |  Update FEP A/B/D matrices       |
          +----------------------------------+
```

### 3.2 VRAM Budget (6GB)

| Component | VRAM | When active |
|---|---|---|
| HALO+FEP (n_tokens=32) | ~1.5 GB | Always |
| JAX XLA overhead | ~0.5 GB | Always |
| Phi-3.5-mini 4-bit | ~2.0 GB | Wake cycle only |
| Embedding models | 0 GB | CPU only |
| FAISS + SQLite | 0 GB | CPU/RAM |
| **Peak (wake cycle)** | **~4.0 GB** | |
| **Idle** | **~2.0 GB** | |

### 3.3 Self-Improvement at Four Timescales

| Timescale | Mechanism | What updates |
|---|---|---|
| Every step (~60s) | FEP variational inference | `swarm_mu` — working beliefs about current topic |
| Every episode | FEP matrix Bayesian updates | A (likelihood), B (transition), D (prior) |
| On wake cycle | LLM goal-setting | C matrix — preferred observations / goals |
| Nightly | LoRA fine-tuning | HALO SSM diagonal matrices + attention projections |

---

## 4. Component Specifications

### 4.1 Perception Pipeline (`halo_fep/perception/`)

Converts raw web content to `(N_tok, d_model)` tensors.

**Files:**
- `halo_fep/perception/__init__.py`
- `halo_fep/perception/web_fetcher.py` — DuckDuckGo search, HTML-to-markdown cleaning
- `halo_fep/perception/embedder.py` — text and image embedding, projection to d_model
- `halo_fep/perception/token_packer.py` — packs results into (n_tokens=32, d_model=256)

**Search:** DuckDuckGo via `duckduckgo-search` Python library (no API key, rate-limited to ~1 req/min).

**Embedding (CPU only):**
- Text: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) → linear projection to d_model=256
- Images: `openai/clip-vit-base-patch32` (512-dim) → linear projection to d_model=256
- Projectors: two `eqx.nn.Linear` layers, trained alongside HALO

**Token layout (n_tokens=32):**
- Tokens 0–3: query embedding (4 tokens, one per search keyword cluster)
- Tokens 4–23: top-5 result title+snippet (4 tokens per result)
- Tokens 24–31: top-5 result images where available (1-2 tokens per result)
- Padding: zero-filled if fewer results

**Interface:**
```python
class PerceptionPipeline:
    def embed(self, query: str) -> jnp.ndarray  # (32, 256)
    def query_from_beliefs(self, carry: HaloFEPCarry) -> str
```

`query_from_beliefs` reads the dominant belief cluster from `swarm_mu` and the current C matrix goal to form a natural-language search query.

---

### 4.2 Episodic Memory (`halo_fep/memory/`)

CPU-based, semantically searchable, unlimited size.

**Files:**
- `halo_fep/memory/__init__.py`
- `halo_fep/memory/episode_store.py` — FAISS + SQLite backend
- `halo_fep/memory/schema.py` — Episode dataclass

**Episode schema:**
```python
@dataclass
class Episode:
    id: str                    # UUID
    timestamp: float           # unix time
    query: str                 # search query that triggered this
    tokens: np.ndarray         # (32, 256) float32 — what HALO processed
    swarm_mu: np.ndarray       # (256, 8) — belief state after processing
    free_energy: float         # FEP free energy scalar
    free_energy_delta: float   # change from previous episode (negative = learned)
    llm_output: str | None     # LLM output if wake cycle triggered
    topic_tags: list[str]      # extracted topic keywords
```

**FAISS index:** `IndexFlatIP` over 256-dim query embeddings (cosine similarity after L2 normalization). SQLite stores full episode data keyed by UUID.

**Interface:**
```python
class EpisodeStore:
    def add(self, episode: Episode) -> None
    def retrieve(self, query_embed: np.ndarray, k: int = 5) -> list[Episode]
    def get_recent(self, n: int = 500) -> list[Episode]
    def get_high_confidence(self, min_delta: float = -0.1) -> list[Episode]
    def rebuild_index(self) -> None  # recovery from corruption
```

---

### 4.3 HALO State Compressor (`halo_fep/intellect/state_compressor.py`)

Deterministic formatter — no neural network. Converts the FEP/HALO carry state to a structured natural-language prompt the LLM can reason about.

**Input:** `carry: HaloFEPCarry`, `recent_memories: list[Episode]`, `current_query: str`, `free_energy: float`

**Output prompt structure:**
```
CURRENT STATE
Query: <current search query>
Surprise level: <low|medium|high|very high> (FE=<scalar>)
Dominant belief: <top belief cluster description>
Dominant action: <argmax of mean swarm_action>

RECENT MEMORY (most similar past episodes)
[1] <timestamp> | <query> | FE delta: <value>
[2] ...

GOAL: <current C matrix interpretation>

What should I do next? Reply with exactly one of:
SEARCH: <new search query>
GOAL: <new goal description>
LEARN: <structured fact to remember>
IDLE
```

The LLM receives exactly this prompt. Its response is parsed by prefix matching on the first token of its reply.

---

### 4.4 LLM Bridge (`halo_fep/intellect/llm_bridge.py`)

On-demand Phi-3.5-mini-instruct integration via `transformers` with 4-bit quantization (`bitsandbytes`).

**Model:** `microsoft/Phi-3.5-mini-instruct` at 4-bit NF4 quantization (~2.0GB VRAM).

**Interface:**
```python
class LLMBridge:
    def load(self) -> None    # load to CUDA, ~3s startup
    def unload(self) -> None  # del model, torch.cuda.empty_cache()
    def think(self, prompt: str, max_tokens: int = 128) -> str
    @property
    def is_loaded(self) -> bool
```

**Load/unload contract:** The bridge is always unloaded between wake cycles. `load()` is called at the start of each wake cycle, `unload()` at the end. This ensures HALO+FEP always has headroom.

**Error handling:** If CUDA OOM occurs during load, reduce `max_tokens` to 64 and retry. If it fails again, log and skip the wake cycle — the heartbeat continues without it.

---

### 4.5 FEP Goal Updater (`halo_fep/intellect/goal_updater.py`)

Translates LLM `GOAL:` output into an update to the FEP C matrix (preferred observations).

**C matrix:** shape `(n_obs=4,)` — a probability distribution over preferred observation categories. Currently uniform (no preference). After a goal is set, it biases EFE action selection toward observations matching the goal.

**Mechanism:**
1. Embed LLM goal text using the same `all-MiniLM-L6-v2` embedder (CPU)
2. Project 384-dim embedding → 4-dim via a learned linear layer (trained alongside HALO)
3. Softmax normalize → new C vector
4. Replace `model.gm.C` via equinox tree update (returns updated model)

**Decay:** C matrix decays toward uniform over 100 steps if not refreshed (`C_new = 0.99 * C + 0.01 * uniform`). This prevents the system from getting stuck on a single goal indefinitely.

---

### 4.6 LoRA Training Loop (`halo_fep/training/lora_trainer.py`)

Nightly fine-tuning of HALO weights on the day's high-confidence episodes.

**What fine-tunes:** The `SimpleSSM` diagonal matrices (A, B, C, D — ~4KB each) and the `HoloAttention` w_q, w_k, w_v projection weights. These are the smallest and most semantically meaningful parameters. The `HALOBackbone` FFN layers and `HoloEmbedding` are frozen.

**High-confidence episodes:** episodes where `free_energy_delta < -0.05` (FEP surprise decreased by at least 5% — the model genuinely learned something).

**Training procedure:**
```python
# 1. Load last 500 episodes from EpisodeStore
# 2. Filter to high-confidence subset
# 3. Build (tokens, swarm_mu) pairs as training data
# 4. Run 100 steps of optax.adam(lr=1e-4) on unified_elbo_loss
# 5. Save checkpoint
# 6. Log: loss_before, loss_after, n_episodes_used
```

**Duration:** ~5 minutes on a 6GB GPU for 100 steps over 500 episodes.

**Safety:** If fine-tuning loss is higher after training than before (divergence), revert to the checkpoint from the previous night.

---

### 4.7 FEP Matrix Updater (`halo_fep/training/fep_updater.py`)

Online Bayesian updates to the discrete generative model matrices after each episode.

**What updates:**
- `A` (likelihood): Observation counts per hidden state → running average update
- `D` (prior): Belief frequency → Dirichlet posterior update
- `B` (transition): Action-to-state transitions → running average update

**When:** After every subconscious tick, using the episode's `swarm_mu` (posterior beliefs) and the observation tokens.

**Constraint:** All matrices remain proper probability distributions (row-stochastic or normalized) after each update.

---

### 4.8 Heartbeat Loop (`halo_fep/main.py`)

The top-level orchestrator.

```python
def main():
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)
    model = load_or_init_checkpoint(cfg)
    carry = model.init_carry(jax.random.PRNGKey(cfg.seed))
    perception = PerceptionPipeline(cfg)
    memory = EpisodeStore(path="data/episodes/")
    compressor = StateCompressor(cfg)
    llm = LLMBridge()
    updater = GoalUpdater(cfg)
    lora = LoRATrainer(cfg)

    log.info("Heartbeat started.")

    while True:
        tick_start = time.time()

        # Subconscious tick
        query = perception.query_from_beliefs(carry)
        try:
            tokens = perception.embed(query)
        except Exception as e:
            log.warning(f"Perception failed: {e}. Skipping tick.")
            time.sleep(cfg.tick_interval)
            continue

        key, carry_key = jax.random.split(carry.key)
        carry, info = halo_fep_step(model, carry, tokens, carry_key)
        fe = float(compute_free_energy(carry, model))

        episode = Episode(query=query, tokens=tokens, swarm_mu=carry.swarm_mu,
                          free_energy=fe, ...)
        memory.add(episode)
        model = fep_updater.update(model, episode)

        log.info(f"Tick | query={query!r} | FE={fe:.3f}")

        # Wake cycle
        if fe > cfg.wake_threshold:
            log.info("Wake cycle triggered.")
            recent = memory.retrieve(perception.embed_query(query), k=5)
            prompt = compressor.compress(carry, recent, query, fe)
            try:
                llm.load()
                output = llm.think(prompt)
                llm.unload()
                model, carry = updater.apply(model, carry, output)
                episode.llm_output = output
                memory.add(episode)  # update with LLM output
                log.info(f"Wake output: {output!r}")
            except Exception as e:
                log.error(f"Wake cycle failed: {e}")
                llm.unload()

        # Nightly training
        if is_nightly_window():
            log.info("Nightly learning cycle starting.")
            lora.run(memory.get_high_confidence())
            log.info("Nightly learning cycle complete.")

        elapsed = time.time() - tick_start
        time.sleep(max(0, cfg.tick_interval - elapsed))
```

---

## 5. Bootstrap Procedure (Phase 0)

The system cannot start with random HALO weights browsing the web. A warm-up phase is required:

1. **Synthetic pre-training:** Run `unified_elbo_loss` training for 5,000 steps on `MultimodalWorld` (already built). This gives HALO a basic sense of token structure.
2. **Synthetic episodes:** Run `trainer.run_episode()` for 100 episodes to populate the initial FAISS memory with non-random episodes.
3. **Save checkpoint:** Save `model`, projector weights, and initial `EpisodeStore`.
4. **Start heartbeat:** Launch `main.py` with warm checkpoint.

The bootstrap training script is `halo_fep/training/bootstrap.py`.

---

## 6. New Files

| File | Purpose |
|---|---|
| `halo_fep/perception/__init__.py` | Package |
| `halo_fep/perception/web_fetcher.py` | DuckDuckGo search + HTML cleaning |
| `halo_fep/perception/embedder.py` | text/image → d_model projectors |
| `halo_fep/perception/token_packer.py` | Pack results into (32, 256) |
| `halo_fep/memory/__init__.py` | Package |
| `halo_fep/memory/episode_store.py` | FAISS + SQLite episodic store |
| `halo_fep/memory/schema.py` | Episode dataclass |
| `halo_fep/intellect/__init__.py` | Package |
| `halo_fep/intellect/state_compressor.py` | HALO state → LLM prompt |
| `halo_fep/intellect/llm_bridge.py` | Phi-3.5-mini on-demand loader |
| `halo_fep/intellect/goal_updater.py` | LLM output → C matrix update |
| `halo_fep/training/lora_trainer.py` | Nightly HALO LoRA fine-tuning |
| `halo_fep/training/fep_updater.py` | Online FEP matrix Bayesian update |
| `halo_fep/training/bootstrap.py` | Phase 0 synthetic pre-training |
| `halo_fep/main.py` | Heartbeat orchestrator |

---

## 7. Unchanged Files

All of the following remain exactly as built:
- `halo_fep/halo_jax/` — entire backbone
- `halo_fep/bridge/` — ObsBridge (softmax), ActionBridge, BeliefBridge
- `halo_fep/loss.py` — unified_elbo_loss
- `halo_fep/model.py` — HaloFEPModel, halo_fep_step
- `fep_swarm/` — entire FEP swarm stack
- `halo_fep/config.py` — extended only (add `n_tokens=32`, `wake_threshold`, `tick_interval`)

---

## 8. New Dependencies

```
duckduckgo-search>=6.0     # web search, no API key
sentence-transformers>=3.0 # all-MiniLM-L6-v2 text embedder
transformers>=4.40         # Phi-3.5-mini-instruct
bitsandbytes>=0.43         # 4-bit NF4 quantization
faiss-cpu>=1.8             # episodic memory index
SQLAlchemy>=2.0            # episode metadata store
Pillow>=10.0               # image loading for CLIP
```

---

## 9. What is Explicitly Not Built

- No GUI or web dashboard (terminal logs + structured JSON log file)
- No external API exposure
- No arbitrary code execution (replaced by LoRA fine-tuning)
- No Llama-3-Hermes-8B (replaced by Phi-3.5-mini)
- No self-restarting process (nightly training runs in-process)
- No cloud dependencies

---

## 10. Success Criteria

| Signal | Target | Failure |
|---|---|---|
| Free energy (30-day moving avg) | Decreasing trend | Flat or increasing |
| Wake cycle frequency (weekly) | Decreasing over months | Constant |
| FAISS store growth | Steady accumulation | Empty or stagnant |
| Nightly loss delta | Negative (improving) | Positive (diverging) |
| LLM goal diversity (monthly) | Varied topics | Same goal repeated |
| Heartbeat uptime | >95% | Frequent crashes |

---

## 11. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| DuckDuckGo rate limiting / IP block | Medium | 60s tick interval, user-agent rotation, exponential backoff |
| Phi-3.5 OOM on wake cycle | Low | Load/unload contract; max_tokens=64 fallback |
| Nightly LoRA diverges | Low | Revert to previous night's checkpoint |
| FAISS index corruption | Low | Rebuild from SQLite on startup if check fails |
| JAX NaN in HALO step | Low | Detect NaN, reload checkpoint, continue |
| Embedding model unavailable | Medium | Cache models locally on first download |
