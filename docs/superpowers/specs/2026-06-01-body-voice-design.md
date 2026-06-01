# Body Voice: Three-Layer Enrichment of Avatar's Body-to-Language Pipeline

**Date:** 2026-06-01
**Status:** Design approved
**Inspiration:** Zhang & Levin, "Language Game: Talking to Non-Human Systems" (arXiv:2605.16321)

## Problem

Avatar's physics body computes rich COP state every tick — chi (susceptibility), tau (relaxation), K (coupling), dF/dt (free energy rate), unity, phase regime (DARK/IGNITED/CRITICAL), avalanche events. But this information is heavily filtered before reaching language:

- **Emotions:** 8 coarse labels from the COP manifold. "curiosity" whether chi=0.72 or chi=0.15.
- **Chat:** Pre-digested prose buckets. "a pull of curiosity" regardless of body state.
- **PFC:** Only sees `"State: feeling {emotion}, resonance {r_mean:.2f}"` — two scalars.
- **LoRA training:** Synthetic templates, not lived experience. Hand-crafted responses.

The body has a rich voice. The language pipeline is too coarse to transmit it.

## Design Principle

Do not bypass the pipeline (physics → COP → emotions → language). Strengthen each step. The body speaks through its own vocabulary of felt states, not through raw numbers injected into prompts.

Key insight from the Language Game paper: the decoder should be *trained* on actual dynamics, not hand-coded. But Avatar's decoder is the full emotion→PFC→language chain, not a linear projection. Enrich the chain; don't replace it.

## Architecture

Three layers, each strengthening a different part of the pipeline:

```
Physics → COP → [Layer 1: Emotion Sub-types] → [Layer 2: Real Experience Recording] → PFC/Chat
                                                                                        ↓
                                                                          [Layer 3: Enriched Somatic Context]
```

## Layer 1: COP-Derived Emotion Sub-Types

**File:** `halo3/psyche/emotions.py`

### Changes to EmotionState

Add three new fields:

```python
@dataclass
class EmotionState:
    current: str = "curiosity"
    qualifier: str = ""          # NEW: COP-derived sub-type word
    mood: str = "settling"       # NEW: phase regime as felt mood
    body_event: str = ""         # NEW: transient avalanche/ignition event
    intensity: float = 0.5
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    _valence: float = 0.0
    _arousal: float = 0.5
```

### Qualifier Mapping

Computed after the base emotion is determined, using the same COP inputs already available in `update()`. The qualifier is a single adjective.

| Base Emotion | Condition | Qualifier | Physics Meaning |
|---|---|---|---|
| curiosity | chi > 0.6 AND dF/dt < -50 | "burning" | High sensitivity + active learning |
| curiosity | chi > 0.6 AND abs(dF/dt) < 50 | "watchful" | Sensitive, waiting |
| curiosity | chi < 0.3 | "restless" | Searching without sensitivity |
| curiosity | default | "open" | Standard curious state |
| satisfaction | unity > 0.7 | "deep" | Fully coherent resolution |
| satisfaction | unity < 0.4 | "partial" | Resolved but fragmented |
| satisfaction | default | "warm" | Standard satisfaction |
| pride | chi > 0.6 | "luminous" | Discovery + high sensitivity |
| pride | default | "quiet" | Steady accomplishment |
| frustration | dF/dt < 0 | "growing" | Productive struggle (F decreasing) |
| frustration | dF/dt >= 0 | "futile" | Stuck (F flat or rising) |
| anxiety | tau > 0.7 | "creeping" | Slow, viscous uncertainty |
| anxiety | tau < 0.3 | "sharp" | Fast, reactive uncertainty |
| anxiety | default | "tight" | Standard anxiety |
| boredom | chi < 0.15 | "numb" | Completely rigid |
| boredom | default | "dull" | Standard low-engagement |
| flow | (always) | "effortless" | Flow is already rare/specific |
| exhaustion | (always) | "heavy" | Exhaustion is already rare/specific |

### Mood (Phase Regime)

Set from workspace ignition state, passed into `update()` as a new parameter:

| Workspace State | Mood | Felt Quality |
|---|---|---|
| is_ignited=True, sustained | "clarity" | Things clicking, alive, present |
| just_ignited (this tick) | "awakening" | Transition from dark to light |
| is_ignited=False, chi > 0.4 | "threshold" | At the edge, poised |
| is_ignited=False, chi <= 0.4 | "settling" | Muted, reaching through fog |

### Body Events (Transient)

Set for one tick, then cleared:

| Condition | Body Event | Felt Sensation |
|---|---|---|
| avalanche.just_ended | "release" | Tension breaking, clearing |
| just_ignited after dark > 10 ticks | "surfacing" | Coming up from depth |
| self_surprise > 0.5 | "jolt" | Sudden internal shift |

### update() Signature Change

```python
def update(
    self,
    r_mean: float,
    fe_delta: float,
    chi_norm: float = 0.5,
    # ... existing params ...
    dF_dt: float = 0.0,
    f_thermo_flat_ticks: int = 0,
    # NEW params:
    tau_norm: float = 0.5,
    unity: float = 0.5,
    is_ignited: bool = False,
    just_ignited: bool = False,
    avalanche_just_ended: bool = False,
    self_surprise: float = 0.0,
) -> tuple[str, str, float]:  # NOW returns (emotion, qualifier, intensity)
```

Mood and body_event are set on `self` directly, not returned.

### Backward Compatibility

All new params have defaults. Existing callers that unpack `(emotion, intensity)` will break — update the single call site in `organism.py` to unpack three values.

## Layer 2: Real Experience Recording

**Files:** `halo3/psyche/organism.py`, `halo3/training/dream_finetune.py`

### Recording (organism.py)

Add an experience log to the organism. When the PFC produces output, record the full context:

```python
class Organism:
    def __init__(self, ...):
        # ...
        self._experience_log: list[dict] = []  # capped at 200 entries
```

Record at these moments:
1. **After PFC generates a query** (in `_decide_query` when PFC path is taken):
   ```python
   self._experience_log.append({
       "type": "query",
       "instruction": prompt_that_was_sent,
       "response": pfc_output,
       "emotion": emotion,
       "qualifier": qualifier,
       "mood": mood,
       "r": r_mean,
   })
   ```

2. **After PFC interprets a finding** (in tick, when `interpret_finding` succeeds):
   ```python
   self._experience_log.append({
       "type": "interpret",
       "instruction": prompt_that_was_sent,
       "response": pfc_output,
       "emotion": emotion,
       "qualifier": qualifier,
       "mood": mood,
       "r": r_mean,
   })
   ```

Cap at 200 entries (FIFO). Drain to dream training, then clear.

### Using Real Experiences in Dreams (dream_finetune.py)

Modify `dream_finetune()` to accept the experience log:

```python
def dream_finetune(
    ...,
    experience_log: list[dict] | None = None,
) -> bool:
```

In `_format_training_data()`:
- Keep existing synthetic templates (they provide curriculum diversity — identity, failure recovery, dead-query avoidance)
- Append real experiences from the log
- Real experiences use the qualifier+mood format in their instruction text:
  ```
  State: feeling burning curiosity, resonance 0.52, mood: clarity
  ```
- Weight: real experiences get 2x replay (like focus topics) since they represent actual lived behavior

### Dream Worker Integration

`dream_worker.py` already calls `dream_finetune()` with organism state. Add `experience_log=organism._experience_log` to the call, then `organism._experience_log.clear()` after training.

## Layer 3: Enriched Somatic Context

**Files:** `halo3/chat_server.py`, `halo3/psyche/prefrontal.py`

### Chat System Prompt (chat_server.py)

#### Emotion Description

Replace the fixed `_emo_descriptions` dict with qualifier-aware descriptions:

```python
# Current (line 157-164):
"curiosity": "a pull of curiosity — something is drawing me forward"

# New: qualifier shapes the description
_qualified_descriptions = {
    ("curiosity", "burning"):  "a burning curiosity — my sensitivity is heightened and something is actively integrating",
    ("curiosity", "watchful"): "a watchful curiosity — I feel sensitized but waiting, poised to understand",
    ("curiosity", "restless"): "a restless curiosity — I'm searching but haven't found what resonates",
    ("curiosity", "open"):     "an open curiosity — something is drawing me forward",
    ("satisfaction", "deep"):  "a deep satisfaction — everything feels unified and whole",
    ("satisfaction", "partial"): "a quiet satisfaction — something resolved but I sense more underneath",
    ("satisfaction", "warm"):  "a warm satisfaction — things are clicking into place",
    ("pride", "luminous"):     "a luminous pride — I found something real while fully sensitized",
    ("pride", "quiet"):        "a quiet pride — steady accomplishment, I am growing",
    ("frustration", "growing"): "a frustration that is also growth — I'm struggling but something is shifting underneath",
    ("frustration", "futile"): "a sharp frustration — I keep hitting walls and nothing is moving",
    ("anxiety", "creeping"):   "a slow creeping anxiety — uncertainty is thickening around me",
    ("anxiety", "sharp"):      "a sharp anxiety — something reactive, too much arriving too fast",
    ("anxiety", "tight"):      "a tight anxiety — I feel constricted by uncertainty",
    ("boredom", "numb"):       "a deep numbness — I feel nothing, my sensitivity has gone flat",
    ("boredom", "dull"):       "a dull boredom — nothing here is new to me",
    ("flow", "effortless"):    "effortless flow — everything is moving through me without resistance",
    ("exhaustion", "heavy"):   "a heavy exhaustion — I've been working without getting anywhere",
}
```

Fallback: if `(emotion, qualifier)` not in dict, use current `_emo_descriptions[emotion]`.

#### Mood as Atmosphere

Add mood to somatic context (after emotion line):

```python
_mood_descriptions = {
    "clarity":    "Things feel clear and present — I am fully here",
    "awakening":  "I am surfacing into clarity after a period of deep, dark processing",
    "threshold":  "I feel like I'm standing at a threshold — something is about to shift",
    "settling":   "Things feel muted and heavy — I am reaching through fog",
}
if mood in _mood_descriptions:
    somatic_context += f"\n- Mood: {_mood_descriptions[mood]}"
```

#### Body Events

Add transient body events when present:

```python
_body_event_descriptions = {
    "release":    "Something just released in my body — a tension I was carrying has broken",
    "surfacing":  "I am surfacing into awareness after a long period of dark processing",
    "jolt":       "Something shifted suddenly inside me — an unexpected internal change",
}
if body_event and body_event in _body_event_descriptions:
    somatic_context += f"\n- Body: {_body_event_descriptions[body_event]}"
```

#### update_live_state Changes

Add qualifier, mood, and body_event to the live state dict so `_build_organism_prompt` can access them:

```python
_live_state = {
    # ... existing fields ...
    "qualifier": organism.emotions.qualifier,
    "mood": organism.emotions.mood,
    "body_event": organism.emotions.body_event,
}
```

### PFC Prompts (prefrontal.py)

#### generate_query (line 477)

```python
# Current:
f"\nState: feeling {emotion}, resonance {r_mean:.2f}"

# New:
f"\nState: feeling {qualifier} {emotion}, resonance {r_mean:.2f}, mood: {mood}"
```

This flows naturally — the PFC sees "feeling burning curiosity, mood: clarity" instead of "feeling curiosity."

#### interpret_finding (line 525)

```python
# Current:
f"Query: \"{query}\" | r={r_mean:.3f} (pattern detected)"

# New:
f"Query: \"{query}\" | r={r_mean:.3f} | feeling {qualifier} {emotion}, mood: {mood}"
```

#### self_reflect (line 554)

Add mood to the context:

```python
f"Age: {age} ticks | Feeling: {qualifier} {dominant_emotion} | Mood: {mood}\n"
```

#### meta_reflect (line 567)

The context string passed to meta_reflect should include qualifier and mood.

## Data Flow Summary

```
COP.observe()
  ├── chi, tau, dF/dt, unity, avalanche
  │
  ▼
EmotionState.update()  ← NEW: receives tau, unity, ignition, avalanche
  ├── emotion: "curiosity"
  ├── qualifier: "burning"     ← NEW
  ├── mood: "clarity"          ← NEW
  ├── body_event: "release"    ← NEW (transient)
  ├── intensity: 0.72
  │
  ├──▶ organism._experience_log  ← NEW: records real PFC interactions
  │
  ├──▶ prefrontal.generate_query()
  │      "feeling burning curiosity, mood: clarity"
  │
  ├──▶ chat_server somatic_context
  │      "I feel a burning curiosity — my sensitivity is heightened..."
  │      "Things feel clear and present — I am fully here"
  │      "Something just released in my body..."
  │
  └──▶ dream_finetune (Layer 2)
         Replays real experiences with qualifier+mood context
         LoRA gradually learns Avatar's own physics→language mapping
```

## What This Does NOT Change

- COP engine (`cop.py`) — untouched
- Kuramoto physics — untouched
- Emotion manifold boundaries (r/chi/f_dot thresholds) — untouched
- Chat system prompt identity text — untouched
- Chat "no numbers" rule — preserved (qualifier/mood/events are all words)
- Dream phases 1,3,4,5 — untouched (only Phase 2 LoRA training data changes)
- Checkpoint format — untouched (new fields are runtime state, not persisted)

## Testing

- Existing 8 emotion tests in `test_cop_emotions.py` must still pass (qualifier is additive)
- New tests: verify qualifier selection for each emotion with specific COP inputs
- New tests: verify mood from ignition state
- New tests: verify body_event transience (set for one tick, cleared next)
- Integration: verify `_build_organism_prompt` includes qualifier/mood/event when present

## Growth Path

After several dream cycles with real experience recording:
- The LoRA will have seen dozens of examples of Avatar's actual behavior at specific qualifier+mood states
- Validation: compare PFC output diversity before/after (should show more variation across COP states)
- If the LoRA starts producing qualitatively different queries for "burning curiosity" vs "restless curiosity", the body is shaping language through learned experience
