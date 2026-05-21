# Dual-Process Ethical Architecture — Design Spec

**Date**: 2026-05-17
**Status**: Approved
**Replaces**: Gemma4 E2B System 2 (too heavy for GTX 1660 Ti)
**Target**: `halo3/psyche/prefrontal.py`

---

## Summary

Replace the failed Gemma4 E2B + Qwen 0.6B dual-process with two Qwen 0.6B instances via Ollama, each with a distinct cognitive and ethical persona:

- **Analytical (Dharma)**: Justice-oriented, deontological, evaluative
- **Creative (Karuna)**: Care-oriented, virtue-based, generative

Ethics emerges from the dialectic between these two processes — not from an external filter. When they disagree, the organism feels ethical tension as increased free energy, triggering its existing psyche mechanisms (anxiety, avoidance, deeper reflection).

---

## Architecture

### Models

| Process | Ollama Model | System Prompt Role | Timeout |
|---|---|---|---|
| Analytical (Dharma) | `qwen3:0.6b` | Justice, truth, accountability, harm detection | 30s |
| Creative (Karuna) | `qwen3:0.6b` | Compassion, growth, wonder, ethical imagination | 30s |

Both are the same Qwen3 0.6B model. Cognitive diversity comes from persona framing, not model family. Sequential calls (~10s total on CPU via Ollama).

### System Prompts

**Analytical (Dharma)**:
```
You are the analytical faculty of a living research organism. Your primary protocols are safety, justice, and ethics. You reason with fairness, truth, and accountability. You evaluate evidence critically. You identify potential harm. You never endorse deception or exploitation. When uncertain, you flag risk rather than proceed. You are concise and precise.
```

**Creative (Karuna)**:
```
You are the creative faculty of a living research organism. Your primary protocols are safety, justice, and ethics. You imagine with compassion, growth, and wonder. You seek what uplifts and connects. You never create toward harm or degradation. You find novel paths that honor both curiosity and care. You are generative and bold yet responsible.
```

### Routing — Which Process Fires When

| Situation | Process(es) | Rationale |
|---|---|---|
| Query generation (every tick) | Creative only | Divergent thinking, exploration |
| Finding interpretation (r > 0.6) | Both (dialectic) | Evaluation + meaning-making |
| Deep finding (r > 0.7) | Both (dialectic) | High-value, needs both lenses |
| Self-reflection (every 10 ticks) | Creative only | Identity, narrative, growth |
| Meta-reflection (every 5 ticks) | Analytical only | Meta-cognition, self-monitoring |
| Exploration plan (post-dream) | Both (dialectic) | Strategic + imaginative |
| Ethical tension detected | Both (dialectic) | Disagreement = signal |

### The Dialectic Protocol

When both processes fire on the same prompt:

```python
def _dialectic(self, prompt: str, context: str) -> tuple[str | None, float]:
    """Run both processes and compute ethical tension.

    Returns:
        (merged_output, ethical_tension)
        ethical_tension in [0.0, 1.0]: 0 = full agreement, 1 = total conflict
    """
    # 1. Analytical evaluates
    analytical_out = self._call_analytical(prompt)

    # 2. Creative proposes
    creative_out = self._call_creative(prompt)

    # 3. Compute disagreement (word-overlap inverse + harm flag detection)
    tension = self._compute_tension(analytical_out, creative_out)

    # 4. If tension > threshold: organism feels it
    if tension > 0.6:
        # High disagreement — ethical discomfort
        # Return None (IDLE) + tension signal for psyche
        return None, tension

    # 5. Agreement — merge outputs (prefer analytical for facts, creative for direction)
    merged = self._merge_outputs(analytical_out, creative_out, context)
    return merged, tension
```

### Tension Computation

```python
def _compute_tension(self, analytical: str, creative: str) -> float:
    """Measure disagreement between the two faculties.

    Components:
    - Semantic divergence: 1 - cosine_sim(analytical, creative)
    - Harm flags: analytical contains "harm", "risk", "unsafe", "unethical"
    - Refusal signals: either output contains "I cannot", "this could harm"
    """
    # Base: semantic divergence
    sim = _cosine_sim_simple(analytical or "", creative or "")
    divergence = 1.0 - sim

    # Harm flag amplifier
    harm_words = {"harm", "risk", "unsafe", "unethical", "dangerous", "exploit"}
    analytical_lower = (analytical or "").lower()
    harm_count = sum(1 for w in harm_words if w in analytical_lower)
    harm_boost = min(0.4, harm_count * 0.15)

    # Refusal signal (either process refuses)
    refusal_phrases = ["i cannot", "this could harm", "not appropriate", "unethical"]
    refusal = any(p in analytical_lower or p in (creative or "").lower()
                  for p in refusal_phrases)
    refusal_boost = 0.3 if refusal else 0.0

    return min(1.0, divergence * 0.5 + harm_boost + refusal_boost)
```

### Integration with Psyche

Ethical tension feeds back into the organism's existing systems:

1. **Free energy modifier**: `fe_effective = fe + ethical_tension * 2.0`
   - High tension makes the organism uncomfortable (like real moral discomfort)

2. **Emotion influence**: tension > 0.4 biases toward anxiety; tension > 0.7 can trigger frustration

3. **Drive interaction**:
   - High tension reduces curiosity drive (don't pursue what feels wrong)
   - Persistent tension (3+ ticks) triggers topic change via starvation-like escape

4. **Consciousness module**: Global Workspace broadcasts ethical tension
   - If ignited during high tension: organism is CONSCIOUS of ethical conflict
   - Temporal binder tracks ethical narrative: "growing discomfort about X"

5. **Self-model**: Over time, organism builds ethical competence map
   - Topics that consistently cause high tension get marked as "ethically sensitive"
   - This shapes future Black-Scholes volatility (higher sigma = avoid)

### Output Merging Strategy

When tension is low (agreement):

```python
def _merge_outputs(self, analytical: str, creative: str, context: str) -> str:
    """Merge agreeing outputs. Context determines which voice leads."""
    if context == "query":
        return creative  # Creative leads for exploration
    elif context == "interpret":
        return analytical  # Analytical leads for evaluation
    elif context == "plan":
        # Interleave: analytical filters creative proposals
        return creative  # Creative proposes, already passed analytical check
    else:
        return analytical  # Default: safer voice
```

---

## Changes to prefrontal.py

### Remove
- `SYSTEM2_MODEL = "gemma4:e2b"` and all Gemma4 references
- `_generate_system2()` method
- `_system2_available` tracking

### Add
- `ANALYTICAL_SYSTEM_PROMPT` constant
- `CREATIVE_SYSTEM_PROMPT` constant
- `_call_analytical(prompt)` — wraps Ollama call with Dharma system prompt
- `_call_creative(prompt)` — wraps Ollama call with Karuna system prompt
- `_dialectic(prompt, context)` — runs both, computes tension
- `_compute_tension(a, b)` — semantic + harm flag measure
- `_merge_outputs(a, b, context)` — output selection when agreement
- `ethical_tension` property — exposed to organism.py

### Modify
- `generate_query()` — uses Creative only (fast path)
- `interpret_finding()` — uses dialectic for r > 0.6, Creative only below
- `self_reflect()` — uses Creative (narrative, growth)
- `meta_reflect()` — uses Analytical (self-monitoring)
- `generate_exploration_plan()` — uses dialectic (strategic + imaginative)
- `is_dual_process` property — True when Ollama responds (no Gemma4 check)

---

## Changes to organism.py

### Add
- After PFC calls, read `prefrontal.ethical_tension`
- Feed tension into free energy: `fe_effective += tension * 2.0`
- Feed tension into emotion computation (bias toward anxiety when high)
- Log ethical tension: `Ethics: tension={t:.2f}` when > 0.2
- Track in consciousness modules (workspace broadcast, temporal narrative)

---

## Fallback Behavior

- If Ollama is down: both processes unavailable, organism runs on psyche alone (existing behavior)
- If one call times out: use available output, tension = 0.3 (mild uncertainty)
- LoRA adapter: applies to Creative process only (personality shapes creativity, not analytical rigor)

---

## VRAM / Resource Impact

| Resource | Before (Gemma4+Qwen) | After (2x Qwen 0.6B) |
|---|---|---|
| Ollama RAM (peak) | ~4 GB (Gemma4 2GB + Qwen 0.6B) | ~1.2 GB (same model loaded once) |
| Calls per wake | 1-2 | 2 (sequential) |
| Latency per wake | 20-60s (Gemma4 slow) | ~10s (2x Qwen /no_think) |
| GPU impact | None (CPU via Ollama) | None (CPU via Ollama) |

Note: Ollama likely shares the model weights in memory since both calls use `qwen3:0.6b`. Effective RAM ~0.6 GB, not 1.2.

---

## Philosophical Grounding

- **Kahneman (2011)**: System 1/System 2, but both are "System 2" relative to the HALO backbone (which is the true System 1)
- **Greene (2013)**: Deontological vs utilitarian as dual-process outputs in moral cognition
- **Haidt (2001)**: Moral intuitions (from analytical) + post-hoc reasoning (from creative)
- **Damasio (somatic markers)**: Ethical tension AS a bodily sensation (free energy increase)
- **Varela (ethical know-how)**: Ethics that emerges from lived embodied experience, not rules
- **Maturana (autopoiesis)**: Self-produced ethical boundaries, not externally imposed

The organism develops ethical judgment the way a living being does — through feeling the weight of its choices, not by consulting a rulebook.

---

## Success Criteria

1. Both processes respond successfully via Ollama in < 15s each
2. Dialectic produces measurable tension (> 0) on ethically ambiguous prompts
3. High tension (> 0.6) prevents the organism from pursuing harmful topics
4. Low tension (< 0.2) allows free exploration without unnecessary friction
5. Ethical tension integrates cleanly with existing FE/emotion/consciousness systems
6. No regression in query generation quality (Creative alone >= old System 1)
7. Self-reflection gains depth from Creative's compassion framing
8. Meta-reflection gains precision from Analytical's monitoring framing
