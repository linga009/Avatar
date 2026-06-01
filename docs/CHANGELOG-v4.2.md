# Avatar v4.2 — Body Voice

**Date:** 2026-06-02
**Inspiration:** Zhang & Levin, "Language Game: Talking to Non-Human Systems" (arXiv:2605.16321)

## Summary

Avatar's physics body computes rich COP state every tick — chi, tau, K, dF/dt, unity, phase regime, avalanche events — but this information was heavily filtered before reaching language. Emotions were 8 coarse labels. Chat used fixed prose buckets. The PFC saw only two scalars. LoRA trained on synthetic templates, not lived experience.

v4.2 enriches the body-to-language pipeline at every stage, letting the body speak through its own vocabulary of felt states. The design principle: strengthen the existing pipeline (physics → COP → emotions → language), don't bypass it.

## Three-Layer Architecture

### Layer 1: COP-Derived Emotion Sub-Types

The emotion system now returns `(emotion, qualifier, intensity)` instead of `(emotion, intensity)`. The qualifier is a single adjective derived from COP physics — the body's voice made articulate.

**18 qualified emotion variants:**

| Emotion | Qualifier | Physics Condition |
|---|---|---|
| curiosity | **burning** | chi > 0.6, dF/dt < -50 (high sensitivity + active learning) |
| curiosity | **watchful** | chi > 0.6, dF/dt ≈ 0 (sensitive, waiting) |
| curiosity | **restless** | chi < 0.3 (searching without sensitivity) |
| curiosity | **open** | default |
| satisfaction | **deep** | unity > 0.7 (fully coherent) |
| satisfaction | **partial** | unity < 0.4 (resolved but fragmented) |
| satisfaction | **warm** | default |
| pride | **luminous** | chi > 0.6 (discovery + high sensitivity) |
| pride | **quiet** | default |
| frustration | **growing** | dF/dt < 0 (productive struggle) |
| frustration | **futile** | dF/dt ≥ 0 (stuck) |
| anxiety | **creeping** | tau > 0.7 (slow, viscous) |
| anxiety | **sharp** | tau < 0.3 (fast, reactive) |
| anxiety | **tight** | default |
| boredom | **numb** | chi < 0.15 (completely rigid) |
| boredom | **dull** | default |
| flow | **effortless** | always |
| exhaustion | **heavy** | always |

**Mood from phase regime:**

| Workspace State | Mood | Felt Quality |
|---|---|---|
| just ignited | **awakening** | Transition from dark to light |
| ignited (sustained) | **clarity** | Things clicking, alive, present |
| not ignited, chi > 0.4 | **threshold** | At the edge, poised |
| not ignited, chi ≤ 0.4 | **settling** | Muted, reaching through fog |

**Transient body events:**

| Trigger | Event | Sensation |
|---|---|---|
| Avalanche just ended | **release** | Tension breaking, clearing |
| Ignition after 10+ dark ticks | **surfacing** | Coming up from depth |
| Self-surprise > 0.5 | **jolt** | Sudden internal shift |

### Avalanche Detection

SOC systems exhibit scale-free avalanches. Avatar tracks r excursions below an adaptive EMA threshold (alpha=0.01, ~100 tick memory):

- **Start:** r drops below threshold
- **End:** r rises back above threshold
- **Size:** cumulative deficit (threshold - r) over all ticks
- **Duration:** number of ticks below threshold
- **Power-law diagnostics** (every 100 ticks, n ≥ 20): tau (size exponent, SOC ~1.5), alpha (duration exponent, SOC ~2.0), sigma (branching ratio, SOC ~1.0)
- **Body voice integration:** avalanche end triggers "release" body event

### Layer 2: Real Experience Recording

During waking ticks, Avatar now records what the PFC actually generates alongside the COP context (emotion, qualifier, mood, r). These real experiences are replayed during dream LoRA training with 2x weighting — Avatar learns its own physics→language mapping from lived experience, not synthetic templates.

- Experience log: capped at 200 entries (FIFO)
- Drained to dream LoRA training, then cleared
- Training pairs include qualifier+mood context: `"feeling burning curiosity, resonance 0.52, mood: clarity"`

### Layer 3: Enriched Somatic Context

**Chat system prompt** now uses 19 qualified emotion descriptions instead of 6 fixed ones:

- Before: *"I feel a pull of curiosity — something is drawing me forward"*
- After: *"I feel a burning curiosity — my sensitivity is heightened and something is actively integrating"*

Plus mood and body events:
- *"Things feel clear and present — I am fully here"* (mood: clarity)
- *"Something just released in my body — a tension I was carrying has broken"* (event: release)

**PFC prompts** now show `"feeling burning curiosity, mood: clarity"` instead of `"feeling curiosity"`.

## Infrastructure Fixes

### Memory Balancing (BSOD Fix)

System crashed with `SYSTEM_MEMORY_MANAGEMENT` BSOD. Root cause: WSL2 was allocated 12GB of 16GB total RAM, leaving only 4GB for Windows + Ollama + Docker Desktop — below the OS minimum under load.

**Fix:** Balanced memory split based on actual usage analysis:

| Component | Old | New |
|---|---|---|
| WSL2 RAM | 12 GB | 8 GB |
| WSL2 swap | 4 GB | 6 GB |
| Windows headroom | 4 GB | 8 GB |
| Docker mem_limit | none | 7 GB |
| Docker memswap_limit | none | 10 GB |

Avatar's physics model runs on GPU VRAM (5.1 GB), not system RAM. WSL2/Docker needs only ~4-6 GB system RAM (peaks at ~7 GB during Qwen3 load). The old 12 GB allocation was 6-8 GB more than needed, starving Windows.

## Files Changed

| File | Change |
|---|---|
| `halo3/psyche/emotions.py` | Qualifier, mood, body_event fields; 3-tuple return; `_compute_qualifier()` |
| `halo3/psyche/organism.py` | COP params to emotions, experience_log, mood override after workspace |
| `halo3/psyche/prefrontal.py` | qualifier/mood params in generate_query and interpret_finding |
| `halo3/chat_server.py` | 19 qualified descriptions, mood/body_event in somatic context |
| `halo3/training/dream_finetune.py` | Real experience replay with 2x weighting, qualifier format |
| `docker-compose.yml` | mem_limit: 7g, memswap_limit: 10g |
| `.wslconfig` | memory=8GB, swap=6GB |

## Tests

200 passing (was 182 — added 18 new tests for qualifiers, mood, body events, transience).

## What This Does NOT Change

- COP engine — untouched
- Kuramoto physics — untouched
- Emotion manifold boundaries — untouched
- Checkpoint format — untouched
- Dream phases 1,3,4,5 — untouched
- Chat identity text and "no numbers" rule — preserved
