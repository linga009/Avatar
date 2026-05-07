# HoloBiont: Phi-3 Removal + Scale-Up Design Spec

**Date:** 2026-05-07
**Status:** Approved
**Scope:** Remove Phi-3.5 LLM dependency; replace with native JAX meta-intelligence; scale model to saturate 6 GB VRAM

---

## 1. Motivation

HoloBiont currently uses `microsoft/Phi-3.5-mini-instruct` (4-bit NF4, ~5.9 GB peak VRAM) as a "System 2" wake cycle that fires when Free Energy exceeds 2.5 nats. This creates three problems:

1. **External dependency** — requires HuggingFace download, `torch`, `transformers`, `bitsandbytes` at runtime
2. **VRAM waste** — the LLM is loaded only on wake events; during normal operation ~4.2 GB of available VRAM sits unused
3. **Architectural mismatch** — an LLM producing natural-language commands is philosophically inconsistent with a FEP organism; goal-setting should emerge from the mathematics itself

**Goals:**
- Remove all LLM runtime dependencies (`torch`, `transformers`, `bitsandbytes`)
- Replace goal-setting with two self-contained JAX mechanisms (Option E + F)
- Scale HALO backbone, swarm, and generative model to utilise ~5.1 GB constantly

---

## 2. VRAM Budget

| Component | Current | Proposed | VRAM (float32) |
|---|---|---|---|
| HALO backbone | d_model=1024, 12 layers, d_ff=4096 | d_model=2048, 20 layers, d_ff=8192 | ~4.0 GB |
| FEP swarm state | 256 agents | 1024 agents | ~negligible |
| MetaLayer (new) | — | d_meta=512, 4 layers, d_ff=2048 | ~50 MB |
| HomeostaticRegulator (new) | — | EMA buffers only | ~negligible |
| Activations + JIT overhead | ~0.6 GB | ~1.0 GB | ~1.0 GB |
| **Total** | **~1.8 GB** | | **~5.1 GB** |
| Headroom | 4.2 GB unused | 0.9 GB headroom | |

Target GPU: NVIDIA RTX with 6 GB VRAM. Headroom absorbs JIT recompilation spikes and swarm vmap temporaries.

---

## 3. Config Changes

All values live in `halo_fep/config.py` (`HaloFEPConfig`).

### 3.1 Scale-Up Parameters

| Parameter | Old | New | Notes |
|---|---|---|---|
| `d_model` | 1024 | 2048 | Hidden dimension |
| `n_heads` | 16 | 16 | Kept; `d_head` becomes 128 |
| `d_head` | 64 | 128 | Auto: d_model / n_heads |
| `n_layers` | 12 | 20 | Deeper backbone |
| `d_state` | 16 | 32 | SSM state dimension |
| `d_ff` | 4096 | 8192 | FFN hidden size |
| `n_agents` | 256 | 1024 | Swarm size |
| `n_hidden` | 8 | 16 | Discrete belief states per agent |
| `n_obs` | 4 | 8 | Observation dimensionality |
| `n_actions` | 4 | 8 | Action space |
| `n_policies` | 8 | 16 | Candidate policies evaluated |
| `coarse_k` | 16 | 32 | Must divide n_agents: 1024 % 32 = 0 ✓ |

Validation in `__post_init__`: `n_heads * d_head == d_model` → 16 * 128 = 2048 ✓

### 3.2 New Meta-Layer Parameters

```python
meta_d_model: int = 512       # MetaLayer hidden dimension
meta_n_layers: int = 4        # MetaLayer depth
meta_d_ff: int = 2048         # MetaLayer FFN size
meta_n_hidden: int = 8        # Meta-belief states
meta_n_obs: int = 16          # Meta-observations (= n_hidden of main model)
meta_n_actions: int = 4       # Meta-action space
meta_k: int = 10              # Ticks between meta-steps
```

### 3.3 New Homeostatic Regulator Parameters

```python
homeo_ema_alpha: float = 0.99           # EMA decay for running mean/var
homeo_novelty_threshold_factor: float = 0.8  # Adaptive threshold = 0.8 * ema_novelty
homeo_blend_clip: float = 1.0           # Max novelty weight before clipping
```

### 3.4 Removed Parameters

```python
# Deleted — no LLM, no wake cycle
wake_threshold: float  # removed
```

---

## 4. New Components

### 4.1 MetaLayer (Option E — Hierarchical FEP)

**File:** `halo_fep/intellect/meta_layer.py`

**Purpose:** A second FEP layer operating at a slow timescale (every K=10 ticks) that deliberates over the organism's recent belief history and sets `log_C` for the main model.

**State (`MetaCarry`):**
```python
@dataclass
class MetaCarry:
    ring_buffer: jnp.ndarray    # (K, n_hidden) — last K mean belief vectors
    ring_idx: int               # current write position
    meta_mu: jnp.ndarray        # (meta_n_hidden,) — current meta-belief
    tick_count: int             # ticks since last meta-step
```

**Generative model (`MetaGenerativeModel`):**
- `A_meta`: `(meta_n_obs=16, meta_n_hidden=8)` — how accumulated belief patterns relate to meta-states
- `B_meta`: `(meta_n_hidden, meta_n_hidden, meta_n_actions=4)` — meta-state transitions
- `D_meta`: `(meta_n_hidden,)` — meta-prior over hidden states
- `C_meta`: `(meta_n_obs,)` — meta-preferences (what belief patterns the organism wants)

**Step logic (`MetaLayer.step`):**
1. Push current `mean_belief` into ring buffer
2. Increment `tick_count`
3. If `tick_count % K != 0`: return unchanged `meta_carry`, `log_C=None`
4. Else:
   a. Reduce ring buffer: `meta_obs = mean(ring_buffer, axis=0)` → `(n_hidden=16,)` = `(meta_n_obs,)` — collapses K belief snapshots into one representative summary
   b. Run variational inference to update `meta_mu` using `meta_obs` as observation (same `belief_update` function as main swarm, applied to meta-GM)
   c. Compute `G_meta` for each of `meta_n_actions` candidate goal vectors
   d. Select goal vector minimising `G_meta`
   e. Return updated `meta_carry`, new `log_C` of shape `(n_obs=8,)`

**Training:** MetaLayer parameters included in `LoRATrainer` trainable mask during nightly dreaming. It learns which belief patterns lead to sustained free-energy reduction.

**Module type:** `eqx.Module` — fully differentiable, JIT-compatible.

---

### 4.2 HomeostaticRegulator (Option F — Novelty-Driven)

**File:** `halo_fep/intellect/homeostatic_regulator.py`

**Purpose:** Fast (every-tick) explore/exploit switch that updates `log_C` based on how novel the current HALO hidden state is relative to recent history.

**State (plain Python + JAX arrays, not eqx.Module — no trainable params):**
```python
h_mean: jnp.ndarray    # (d_model,) running EMA of hidden state mean
h_var: jnp.ndarray     # (d_model,) running EMA of squared deviation
novelty_ema: float     # scalar EMA of recent novelty scores
```

**Novelty score:**
```
novelty = mean( (h_out_mean - h_mean)² / (h_var + ε) )
```
Normalised Mahalanobis-style distance. High when observation is genuinely new.

**Update logic (`HomeostaticRegulator.update(h_out) -> (novelty, log_C_homeo)`):**
1. Compute `h_out_mean = mean(h_out, axis=0)` — `(d_model,)`
2. Compute `novelty` score
3. Update `h_mean`, `h_var`, `novelty_ema` via EMA
4. Compute adaptive threshold: `threshold = homeo_novelty_threshold_factor * novelty_ema`
5. If `novelty > threshold` (explore): `log_C_homeo = log-uniform(n_obs)` — equal preference
6. If `novelty ≤ threshold` (exploit): `log_C_homeo = best_cluster_log_C` derived from recent episode history (lowest mean `free_energy_delta` per observation cluster)
7. Return `(novelty, log_C_homeo)`

---

### 4.3 log_C Blending

After both mechanisms produce a `log_C` candidate each tick, they are blended in `main.py`:

```python
novelty_weight = jnp.clip(novelty / (novelty_ema + 1e-8), 0.0, cfg.homeo_blend_clip)
novelty_weight_norm = novelty_weight / (novelty_weight + 1.0)  # sigmoid-like [0, 1]

if log_C_meta is not None:
    log_C_final = novelty_weight_norm * log_C_homeo + (1 - novelty_weight_norm) * log_C_meta
else:
    log_C_final = log_C_homeo

model = eqx.tree_at(lambda m: m.gm.log_C, model, log_C_final)
```

- High novelty (surprising) → homeostatic regulator dominates → explore
- Low novelty (familiar) + meta-step fired → meta-layer dominates → deliberate goal
- Low novelty + no meta-step → homeostatic exploit mode holds

---

## 5. Modified Components

### 5.1 `halo_fep/model.py`

`HaloFEPCarry` gains a `MetaCarry` field:
```python
class HaloFEPCarry(NamedTuple):
    swarm_mu:     jnp.ndarray   # (n_agents, n_hidden)
    swarm_action: jnp.ndarray   # (n_agents, n_actions)
    page_mem:     PageMemState
    key:          jnp.ndarray   # PRNGKey
    meta_carry:   MetaCarry     # NEW
```

`HaloFEPModel` gains a `MetaLayer` field:
```python
class HaloFEPModel(eqx.Module):
    backbone:   HALOBackbone
    gm:         DiscreteGenerativeModel
    # bridges...
    meta_layer: MetaLayer       # NEW
```

### 5.2 `halo_fep/main.py`

- Remove `LLMBridge`, `StateCompressor` imports
- Add `HomeostaticRegulator`, `MetaLayer` to `HeartbeatLoop.__init__`
- Remove `_wake_cycle` method entirely
- Add `HomeostaticRegulator.update()` call after HALO step
- Add `MetaLayer.step()` call and `log_C` blending
- Remove `wake_threshold` check

### 5.3 `halo_fep/intellect/goal_updater.py`

- Remove text embedding logic (sentence-transformers import, `_proj` matrix, `update_goal` method)
- Keep only `decay()` method — goal decay is still needed every tick

### 5.4 `halo_fep/training/lora_trainer.py`

Extend trainable mask to include `MetaLayer` parameters:
```python
return any(sub in name for sub in [
    'ssm.diag', 'attn.Q', 'attn.K', 'attn.V',
    'meta_layer',   # NEW
])
```

---

## 6. Deleted Files

| File | Reason |
|---|---|
| `halo_fep/intellect/llm_bridge.py` | Phi-3 gone |
| `halo_fep/intellect/state_compressor.py` | Only existed to format LLM prompts |

---

## 7. Dependency Changes

**`requirements.txt` / `pyproject.toml`:**

```
REMOVE:
  torch
  transformers
  bitsandbytes

KEEP:
  jax[cuda]
  equinox
  optax
  faiss-gpu
  sentence-transformers   (still used by perception embedder)
  duckduckgo-search
```

---

## 8. Testing Strategy

### 8.1 New Unit Tests

**`halo_fep/tests/test_meta_layer.py`**
- Init with small config (meta_n_hidden=4, K=3, n_obs=4)
- Feed K synthetic belief vectors → assert ring buffer fills correctly
- Assert `log_C` output shape `(n_obs,)`, no NaN/Inf, valid log-probs (all ≤ 0)
- Assert meta-step only fires every K ticks (not every tick)
- Assert `meta_mu` changes after a meta-step

**`halo_fep/tests/test_homeostatic_regulator.py`**
- Identical hidden states repeated → novelty → 0 → exploit mode
- Random hidden states → high novelty → explore mode → `log_C` uniform
- Assert EMA buffers update (not frozen)
- Assert blend output shape matches `(n_obs,)`

### 8.2 Extended Integration Test

In `halo_fep/tests/test_integration.py`:
- Run 15 ticks with mocked perception
- Assert `MetaLayer` fires exactly once (at tick 10)
- Assert `model.gm.log_C` changes after tick 10
- Assert no `torch`, `transformers`, `bitsandbytes` imports anywhere in execution path

### 8.3 Extended Config Test

In `halo_fep/tests/test_config.py`:
- Assert new params pass `__post_init__`: `16 * 128 == 2048`, `1024 % 32 == 0`
- Assert meta params validate: `meta_k >= 1`, `meta_n_obs == n_hidden`

---

## 9. Data Flow Summary

```
Every tick:
  Perception.embed(query) → tokens (32, 2048)
  halo_fep_step(model, carry, tokens) → carry, (h_out, soft_obs, v_pred, v_target)
  HomeostaticRegulator.update(h_out) → novelty, log_C_homeo
  MetaLayer.step(meta_carry, mean_belief, fe) → meta_carry, log_C_meta (or None)
  blend(log_C_homeo, log_C_meta, novelty) → log_C_final → model.gm.log_C
  FEPUpdater.update(model, carry, episode, soft_obs)
  GoalUpdater.decay(model)
  EpisodeStore.add(episode)

Every K=10 ticks:
  MetaLayer fires, log_C_meta is not None → meta-layer dominates blend

Nightly 02:00-02:15:
  LoRATrainer.run(model, episodes) — trains backbone + MetaLayer jointly
```

---

## 10. What is NOT Changing

- Perception pipeline (WebFetcher, Embedder, TokenPacker)
- EpisodeStore (SQLite + FAISS)
- FEPUpdater (EMA updates to A, B, D matrices)
- LoRATrainer protocol (EWC, PER, revert-on-diverge)
- Nightly dreaming schedule (02:00-02:15)
- PageCurveMemory
- HoloEmbedding, HALOBackbone, AdS-KG prior

---

## 11. Before / After Summary

| Dimension | Before | After |
|---|---|---|
| System 2 | Phi-3.5-mini (5.9 GB, external) | MetaLayer + HomeoReg (~50 MB, JAX) |
| VRAM usage | 1.8 GB base + 5.9 GB spike | ~5.1 GB constant |
| n_agents | 256 | 1024 |
| d_model | 1024 | 2048 |
| n_layers | 12 | 20 |
| d_ff | 4096 | 8192 |
| n_hidden | 8 | 16 |
| n_obs | 4 | 8 |
| n_actions | 4 | 8 |
| External LLM deps | torch, transformers, bitsandbytes | none |
| Goal-setting | Phi-3 text output (not differentiable) | MetaLayer EFE + homeostatic blend (differentiable, trained) |
| Wake latency | 850 ms | 0 ms (no wake cycle) |
