# Avatar 3.0 Cognitive Fixes — Design Spec

**Date:** 2026-05-16
**Status:** Complete
**Problem:** Organism stuck in query loop after dream — "Amentoflavone ginkg 5281600" repeated indefinitely with zero search results.

## Root Cause Analysis

### Primary Bug: LoRA Format Mismatch
- Training format: `### Instruction:\n{x}\n### Response:\n{y}`
- Inference format: `/no_think You are a research organism...`
- The adapter generates garbage that passes cleanup filters

### Cascading Failures
1. **Ollama fallback blocked** — `_generate_local()` always returns non-empty text
2. **Weakness threshold unreachable** — EMA alpha=0.95 keeps competence >0.45, threshold is 0.3
3. **Optimizer state lost after dreams** — PredictiveProcessor._opt_state recreated fresh
4. **Carry state destroyed after dreams** — `init_carry()` zeroes all Kuramoto phases
5. **GEPA module not in Docker image** — dream_gepa.py exists locally but was added after image build
6. **No zero-result detection** — organism processes nothing as if it were normal input

## Fix Categories

### A. Critical Bugs
- A1: Match LoRA training format to inference format (use same `/no_think` template)
- A2: Add quality gating — if local model output is low quality, fall through to Ollama
- A3: Lower weakness threshold from 0.3 to 0.48 + add competence decay
- A4: Persist PredictiveProcessor state; adaptive learning rate
- A5: Warm-start carry after dreams (blend pre-dream carry with fresh)
- A6: Ensure dream_gepa.py ships in Docker image (already exists locally)

### B. Perception & Information Flow
- B1: Track consecutive_zero_results; force topic change after 3 failures
- B2: Query reformulation on sparse results
- B4: Use episodic memory as second perception channel

### C. PFC & Query Generation
- C1: Semantic dedup — reject PFC output if cosine sim > 0.85 vs recent queries
- C2: Multi-candidate generation at different temperatures
- C3: Temperature modulation by emotion
- C4: Pass failure context ("last 5 searches returned nothing")

### D. Emotion System
- D1: Frustration emotion — repeated failure triggers drastic topic change
- D2: Surprise recalibration for zero input
- D3: Emotional inertia (exponential smoothing)

### E. Drive System
- E1: Information starvation emergency override
- E2: Novelty drive (distinct from curiosity)
- E3: Exploration budget tracking

### F. Self-Model
- F1: Competence decay for unvisited topics
- F2: Topic merging by similarity
- F3: Meta-cognitive query success tracking

### H. Dream / Learning
- H1: Fix LoRA curriculum — format-matched, diverse examples including failures
- H2: Post-dream exploration plan
- H5: Validate adapter before deployment

### I. Memory
- I1: Negative memory — track dead queries
- I2: Memory-informed query generation

## Files Modified
- `halo3/psyche/prefrontal.py` — Format fix, quality gating, semantic dedup
- `halo3/psyche/organism.py` — Zero-result tracking, frustration handling, exploration plan
- `halo3/psyche/emotions.py` — Frustration, surprise recalibration, inertia
- `halo3/psyche/drives.py` — Starvation, novelty drive
- `halo3/psyche/self_model.py` — Threshold fix, decay, merging
- `halo3/main.py` — Zero-result detection, carry preservation
- `halo3/training/dream_finetune.py` — Curriculum fix, validation
- `halo3/memory/episode_store.py` — Negative memory, dead query tracking
- `halo3/predictive.py` — State persistence, adaptive lr
