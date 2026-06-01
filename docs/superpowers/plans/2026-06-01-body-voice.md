# Body Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich Avatar's body-to-language pipeline so COP physics shapes emotion qualifiers, felt moods, body events, and LoRA training data — letting the body speak through words, not numbers.

**Architecture:** Three layers strengthen the existing pipeline without bypassing it. Layer 1 adds COP-derived qualifiers/mood/body-events to the emotion system. Layer 2 records real PFC interactions for dream LoRA training. Layer 3 wires qualifiers/mood/events into chat and PFC prompts.

**Tech Stack:** Python, dataclasses, existing EmotionState/Organism/PrefrontalCortex/chat_server

---

### Task 1: Add qualifier, mood, body_event to EmotionState

**Files:**
- Modify: `halo3/psyche/emotions.py`
- Test: `halo3/tests/test_cop_emotions.py`

- [ ] **Step 1: Write failing tests for qualifier, mood, body_event**

Add to `halo3/tests/test_cop_emotions.py`:

```python
def test_emotion_returns_triple():
    """update() now returns (emotion, qualifier, intensity)."""
    es = EmotionState()
    emotion, qualifier, intensity = es.update(r_mean=0.5, fe_delta=-0.01, chi_norm=0.5)
    assert isinstance(emotion, str)
    assert isinstance(qualifier, str)
    assert isinstance(intensity, float)


def test_curiosity_qualifier_burning():
    """High chi + negative dF/dt → burning curiosity."""
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7, dF_dt=-200.0)
    emotion, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.7, dF_dt=-200.0)
    assert emotion == "curiosity"
    assert qualifier == "burning"


def test_curiosity_qualifier_restless():
    """Low chi → restless curiosity."""
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.2)
    emotion, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.2)
    assert emotion == "curiosity"
    assert qualifier == "restless"


def test_satisfaction_qualifier_deep():
    """High unity → deep satisfaction."""
    es = EmotionState()
    for _ in range(5):
        es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2, unity=0.8)
    _, qualifier, _ = es.update(r_mean=0.7, fe_delta=-0.05, chi_norm=0.2, unity=0.8)
    assert qualifier == "deep"


def test_mood_from_ignition():
    """is_ignited=True → mood 'clarity'."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, is_ignited=True)
    assert es.mood == "clarity"


def test_mood_threshold():
    """Not ignited but chi > 0.4 → mood 'threshold'."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, is_ignited=False)
    assert es.mood == "threshold"


def test_body_event_release():
    """Avalanche just ended → body_event 'release'."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, avalanche_just_ended=True)
    assert es.body_event == "release"


def test_body_event_clears():
    """Body event is transient — cleared on next update."""
    es = EmotionState()
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, avalanche_just_ended=True)
    assert es.body_event == "release"
    es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5, avalanche_just_ended=False)
    assert es.body_event == ""


def test_frustration_qualifier_growing():
    """dF/dt < 0 during frustration → 'growing'."""
    es = EmotionState()
    _, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
                                consecutive_failures=5, dF_dt=-100.0)
    assert qualifier == "growing"


def test_frustration_qualifier_futile():
    """dF/dt >= 0 during frustration → 'futile'."""
    es = EmotionState()
    _, qualifier, _ = es.update(r_mean=0.5, fe_delta=0.0, chi_norm=0.5,
                                consecutive_failures=5, dF_dt=50.0)
    assert qualifier == "futile"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_cop_emotions.py -v`
Expected: All new tests FAIL (update() returns 2-tuple, not 3-tuple; no qualifier/mood/body_event fields).

- [ ] **Step 3: Implement qualifier, mood, body_event in EmotionState**

Replace entire `halo3/psyche/emotions.py` with:

```python
"""Emotions — affect derived from COP phase-diagram geometry.

Maps (r, chi_norm, f_dot) to felt emotional states with COP-derived qualifiers.

COP replaces the if/elif threshold tree with manifold position:
  High r + low chi + resolving -> satisfaction (ordered, calm)
  High r + high chi + resolving -> pride (ordered + sensitive)
  Mid r + high chi -> curiosity (at the critical edge)
  Low r + low chi -> boredom (disordered, rigid)
  Low r + high chi + worsening -> anxiety (disordered, reactive)
  Sustained failure -> frustration (punches through)

Qualifiers enrich each emotion with COP sub-type (e.g. "burning" curiosity).
Mood reflects phase regime (DARK/IGNITED/CRITICAL) as felt atmosphere.
Body events are transient felt sensations from avalanches/ignition.

Emotional inertia (valence/arousal EMA) preserved from v3.11.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque


EMOTION_NAMES = ("satisfaction", "pride", "curiosity", "boredom", "anxiety",
                 "frustration", "flow", "exhaustion")


@dataclass
class EmotionState:
    """Tracks current emotion and emotional history."""
    current: str = "curiosity"
    qualifier: str = ""
    mood: str = "settling"
    body_event: str = ""
    intensity: float = 0.5
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    _valence: float = 0.0
    _arousal: float = 0.5

    def update(
        self,
        r_mean: float,
        fe_delta: float,
        chi_norm: float = 0.5,
        perception_failed: bool = False,
        consecutive_failures: int = 0,
        sensory_novelty: float = 0.0,
        sensory_stability: int = 0,
        speech_detected: bool = False,
        dF_dt: float = 0.0,
        f_thermo_flat_ticks: int = 0,
        tau_norm: float = 0.5,
        unity: float = 0.5,
        is_ignited: bool = False,
        just_ignited: bool = False,
        avalanche_just_ended: bool = False,
        self_surprise: float = 0.0,
    ) -> tuple[str, str, float]:
        """Compute emotion from COP phase-diagram position.

        Returns (emotion_name, qualifier, intensity).
        """
        # Clear transient body event from previous tick
        self.body_event = ""

        f_dot = -fe_delta  # positive when surprise is resolving

        # Sensory novelty amplifies openness
        effective_chi = min(1.0, chi_norm + 0.15 * sensory_novelty
                           if sensory_novelty > 0.8 else chi_norm)

        # --- Metabolic flow: highest priority when conditions met ---
        if effective_chi > 0.3 and dF_dt < -100.0:
            emotion = "flow"
            intensity = min(1.0, effective_chi * 0.5 + min(1.0, abs(dF_dt) / 5000.0) * 0.5)
        elif f_thermo_flat_ticks >= 20 and effective_chi > 0.2:
            emotion = "exhaustion"
            intensity = min(1.0, 0.4 + f_thermo_flat_ticks * 0.02)
        elif consecutive_failures >= 3 or (r_mean < 0.2 and consecutive_failures >= 2):
            if dF_dt < 0:
                emotion = "frustration"
                intensity = min(1.0, 0.4 + consecutive_failures * 0.08)
            else:
                emotion = "frustration"
                intensity = min(1.0, 0.6 + consecutive_failures * 0.1)
        elif r_mean > 0.55 and effective_chi < 0.4 and f_dot > 0.005:
            emotion = "satisfaction"
            intensity = min(1.0, r_mean * (1.0 - effective_chi))
        elif r_mean > 0.55 and effective_chi >= 0.4 and f_dot > 0.005:
            emotion = "pride"
            intensity = min(1.0, r_mean * effective_chi)
        elif r_mean < 0.35 and effective_chi > 0.5 and f_dot < -0.005:
            emotion = "anxiety"
            intensity = min(1.0, effective_chi * (1.0 - r_mean))
        elif r_mean < 0.35 and effective_chi < 0.3:
            emotion = "boredom"
            intensity = max(0.1, 1.0 - effective_chi - r_mean)
        else:
            emotion = "curiosity"
            edge_factor = 1.0 - abs(r_mean - 0.5) * 2.0
            intensity = min(1.0, effective_chi * 0.6 + edge_factor * 0.4)

        # --- Qualifier: COP-derived sub-type ---
        qualifier = self._compute_qualifier(emotion, effective_chi, dF_dt, tau_norm, unity)

        # --- Mood: phase regime as felt atmosphere ---
        if just_ignited:
            self.mood = "awakening"
        elif is_ignited:
            self.mood = "clarity"
        elif effective_chi > 0.4:
            self.mood = "threshold"
        else:
            self.mood = "settling"

        # --- Body events: transient felt sensations ---
        if avalanche_just_ended:
            self.body_event = "release"
        elif just_ignited:
            self.body_event = "surfacing"
        elif self_surprise > 0.5:
            self.body_event = "jolt"

        # --- Emotional inertia (EMA smoothing) ---
        _emo_va = {
            "satisfaction": (0.7, 0.2),
            "pride":        (0.9, 0.8),
            "curiosity":    (0.3, 0.6),
            "boredom":      (-0.3, 0.1),
            "anxiety":      (-0.6, 0.9),
            "frustration":  (-0.8, 0.8),
            "flow":         (0.8, 0.7),
            "exhaustion":   (-0.1, 0.15),
        }
        new_v, new_a = _emo_va.get(emotion, (0.0, 0.5))
        alpha = 0.6
        self._valence = alpha * new_v + (1.0 - alpha) * self._valence
        self._arousal = alpha * new_a + (1.0 - alpha) * self._arousal

        if sensory_stability > 3:
            self._arousal *= 0.9
        if speech_detected:
            self._valence = min(1.0, self._valence + 0.05)

        smoothed = self._emotion_from_va(self._valence, self._arousal)
        if smoothed != emotion and emotion != "frustration":
            emotion = smoothed
            # Recompute qualifier for smoothed emotion
            qualifier = self._compute_qualifier(emotion, effective_chi, dF_dt, tau_norm, unity)

        self.current = emotion
        self.qualifier = qualifier
        self.intensity = intensity
        self.history.append((emotion, intensity))
        return emotion, qualifier, intensity

    @staticmethod
    def _compute_qualifier(emotion: str, chi: float, dF_dt: float,
                           tau: float, unity: float) -> str:
        if emotion == "curiosity":
            if chi > 0.6 and dF_dt < -50:
                return "burning"
            elif chi > 0.6:
                return "watchful"
            elif chi < 0.3:
                return "restless"
            return "open"
        elif emotion == "satisfaction":
            if unity > 0.7:
                return "deep"
            elif unity < 0.4:
                return "partial"
            return "warm"
        elif emotion == "pride":
            return "luminous" if chi > 0.6 else "quiet"
        elif emotion == "frustration":
            return "growing" if dF_dt < 0 else "futile"
        elif emotion == "anxiety":
            if tau > 0.7:
                return "creeping"
            elif tau < 0.3:
                return "sharp"
            return "tight"
        elif emotion == "boredom":
            return "numb" if chi < 0.15 else "dull"
        elif emotion == "flow":
            return "effortless"
        elif emotion == "exhaustion":
            return "heavy"
        return ""

    @staticmethod
    def _emotion_from_va(valence: float, arousal: float) -> str:
        if valence < -0.5 and arousal > 0.5:
            return "frustration" if valence < -0.7 else "anxiety"
        if valence < -0.1 and arousal < 0.3:
            return "boredom"
        if valence > 0.5 and arousal < 0.4:
            return "satisfaction"
        if valence > 0.5 and arousal >= 0.4:
            return "pride"
        return "curiosity"

    @property
    def dominant_recent(self) -> str:
        if len(self.history) < 2:
            return self.current
        recent = list(self.history)[-20:]
        counts = {}
        for e, _ in recent:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def emoji(self) -> str:
        return {
            "satisfaction": "\U0001f60c",
            "pride": "\u2728",
            "curiosity": "\U0001f50d",
            "boredom": "\U0001f610",
            "anxiety": "\u26a1",
            "frustration": "\U0001f624",
            "flow": "\U0001f525",
            "exhaustion": "\U0001f6b6",
        }.get(self.current, "?")
```

- [ ] **Step 4: Fix existing tests to unpack 3-tuple**

In `halo3/tests/test_cop_emotions.py`, update every existing test that unpacks `(emotion, intensity)` to unpack `(emotion, _qualifier, intensity)` or `(emotion, _, intensity)`:

- `test_emotion_returns_tuple` (line 6-9): change to unpack 3 values and assert qualifier is str
- `test_emotion_labels_valid` (line 14-18): change `emotion, _ =` to `emotion, _, _ =`
- `test_high_r_low_chi_resolving_is_satisfaction` (lines 22-26): change `emotion, _ =` to `emotion, _, _ =`
- `test_high_r_high_chi_resolving_is_pride` (lines 30-34): change `emotion, _ =` to `emotion, _, _ =`
- `test_mid_r_is_curiosity` (lines 38-42): change `emotion, _ =` to `emotion, _, _ =`
- `test_frustration_overrides` (line 47): change `emotion, _ =` to `emotion, _, _ =`
- `test_intensity_range` (line 55): change `_, intensity =` to `_, _, intensity =`
- `test_sensory_novelty_amplifies` (lines 61-63): change `_, i1 =` to `_, _, i1 =` and `_, i2 =` to `_, _, i2 =`

- [ ] **Step 5: Run all emotion tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_cop_emotions.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 6: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/emotions.py halo3/tests/test_cop_emotions.py
git commit -m "feat(psyche): add COP-derived emotion qualifiers, mood, and body events

Layer 1 of body-voice enrichment. EmotionState.update() now returns
(emotion, qualifier, intensity) and sets mood + body_event on self.
Qualifiers: burning/watchful/restless/open curiosity, deep/partial/warm
satisfaction, growing/futile frustration, etc. Mood from phase regime.
Body events from avalanches, ignition, self-surprise."
```

---

### Task 2: Wire qualifier/mood/body_event through organism.py

**Files:**
- Modify: `halo3/psyche/organism.py:55,160-170,193,228,304,426-431,599-614,651-661`

- [ ] **Step 1: Add experience_log to __init__**

In `halo3/psyche/organism.py`, after line 75 (`self._exploration_plan`), add:

```python
        self._experience_log: list[dict] = []  # real PFC interactions for dream LoRA
```

- [ ] **Step 2: Update emotions.update() call to unpack 3-tuple and pass new params**

Replace lines 159-170 (the `emotions.update()` call). Note: `ws` (workspace) is computed AFTER emotions (line 206), so pass `is_ignited`/`just_ignited`/`self_surprise` at their defaults — mood and body_event will be overridden after workspace in Step 2b.

```python
        # 2. Compute emotion (with failure context + sensory signals + thermodynamics)
        emotion, qualifier, intensity = self.emotions.update(
            r_mean, fe_delta,
            chi_norm=chi_norm,
            perception_failed=perception_failed,
            consecutive_failures=self._consecutive_zero_results,
            sensory_novelty=sensory_novelty,
            sensory_stability=sensory_stability,
            speech_detected=speech_detected,
            dF_dt=_dF_dt,
            f_thermo_flat_ticks=self._f_thermo_flat_ticks,
            tau_norm=tau_norm,
            unity=cop["unity"],
            avalanche_just_ended=cop.get("avalanche", {}).get("just_ended", False),
        )
```

Then AFTER the workspace update (after line ~211) and introspection (line ~182), add:

```python
        # Update mood and body_event now that workspace and introspection are computed
        if ws["just_ignited"]:
            self.emotions.mood = "awakening"
            if ws.get("dark_duration_before", 0) > 10:
                self.emotions.body_event = "surfacing"
        elif ws["is_ignited"]:
            self.emotions.mood = "clarity"
        elif chi_norm > 0.4:
            self.emotions.mood = "threshold"
        else:
            self.emotions.mood = "settling"

        if self_surprise > 0.5 and not self.emotions.body_event:
            self.emotions.body_event = "jolt"
```

Place this block after line 213 (`log.info(f"  ★ IGNITION...")`), before meditation (line 216).

- [ ] **Step 3: Pass qualifier and mood to PFC generate_query**

At line 426-431, update the `generate_query` call to pass qualifier and mood:

```python
        pfc_query = self.prefrontal.generate_query(
            current_query, emotion, r_mean, texts,
            self.self_model.strengths,
            consecutive_failures=self._consecutive_zero_results,
            dead_queries=self.self_model.dead_queries,
            qualifier=self.emotions.qualifier,
            mood=self.emotions.mood,
        )
```

- [ ] **Step 4: Record real PFC experience after query generation**

After line 434 (`return pfc_query`), add experience recording. Actually, since it returns, record before the return:

Replace lines 432-434:

```python
        if pfc_query:
            log.debug(f"Prefrontal: generated query '{pfc_query}'")
            if len(self._experience_log) < 200:
                self._experience_log.append({
                    "type": "query",
                    "response": pfc_query,
                    "emotion": emotion,
                    "qualifier": self.emotions.qualifier,
                    "mood": self.emotions.mood,
                    "r": r_mean,
                })
            return pfc_query
```

- [ ] **Step 5: Pass qualifier/mood to interpret_finding and record experience**

At line 193, update the `interpret_finding` call to pass qualifier and mood:

```python
            pfc_finding = self.prefrontal.interpret_finding(
                texts, current_query, r_mean,
                qualifier=self.emotions.qualifier,
                emotion=emotion,
                mood=self.emotions.mood,
            )
```

After line 194, record the experience if interpret_finding succeeded:

```python
            if pfc_finding and len(self._experience_log) < 200:
                self._experience_log.append({
                    "type": "interpret",
                    "response": pfc_finding,
                    "emotion": emotion,
                    "qualifier": self.emotions.qualifier,
                    "mood": self.emotions.mood,
                    "r": r_mean,
                })
```

- [ ] **Step 6: Pass qualifier/mood to meta_reflect context**

At line 599, append qualifier and mood to the context string:

```python
        context = (
            f"Temporal flow: {thread}. "
            f"Momentum: {momentum}. "
            f"Coherence: {coherence:.2f}. "
            f"Feeling: {self.emotions.qualifier} {self.emotions.current}. "
            f"Mood: {self.emotions.mood}. "
        )
```

- [ ] **Step 7: Pass experience_log to dream_finetune**

At line 651, add `experience_log` to the `dream_finetune()` call:

```python
            success = dream_finetune(
                age=self.self_model.age,
                competence=self.self_model.competence,
                traits=self.self_model.traits,
                narrative=self.self_model.narrative,
                strengths=self.self_model.strengths,
                weaknesses=self.self_model.weaknesses,
                findings=findings,
                dead_queries=self.self_model.dead_queries,
                focus_topics=focus_topics,
                experience_log=self._experience_log,
            )
```

After line 665 (`self.temporal.reset_focus()`), clear the experience log:

```python
            self._experience_log.clear()
```

- [ ] **Step 8: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/organism.py
git commit -m "feat(psyche): wire qualifier/mood/body_event through organism

Passes COP-derived emotion enrichments to PFC, records real PFC
interactions in experience_log for dream LoRA training. Mood set
after workspace ignition (not in emotions.update) to avoid
circular dependency."
```

---

### Task 3: Update PFC prompts with qualifier and mood

**Files:**
- Modify: `halo3/psyche/prefrontal.py:434-443,477,523-528,554-558`

- [ ] **Step 1: Add qualifier and mood params to generate_query**

At line 434-443 of `prefrontal.py`, add `qualifier` and `mood` params:

```python
    def generate_query(
        self,
        current_query: str,
        emotion: str,
        r_mean: float,
        texts: list[str],
        strengths: list[str],
        consecutive_failures: int = 0,
        dead_queries: list[str] | None = None,
        qualifier: str = "",
        mood: str = "",
    ) -> str | None:
```

- [ ] **Step 2: Update the State line in generate_query prompt**

Replace line 477:

```python
            f"\nState: feeling {emotion}, resonance {r_mean:.2f}",
```

With:

```python
            f"\nState: feeling {qualifier + ' ' if qualifier else ''}{emotion}, resonance {r_mean:.2f}" + (f", mood: {mood}" if mood else ""),
```

- [ ] **Step 3: Add qualifier/emotion/mood params to interpret_finding**

At line 518-519, update signature:

```python
    def interpret_finding(self, texts, query, r_mean,
                          qualifier: str = "", emotion: str = "", mood: str = "") -> str | None:
```

Replace line 525:

```python
            f"Query: \"{query}\" | r={r_mean:.3f} (pattern detected)\n"
```

With:

```python
            f"Query: \"{query}\" | r={r_mean:.3f} | feeling {qualifier + ' ' if qualifier else ''}{emotion}" + (f", mood: {mood}" if mood else "") + "\n"
```

- [ ] **Step 4: Update self_reflect to include qualifier and mood**

In `self_reflect` (line 554-558), replace the prompt construction:

```python
        prompt = (
            f"{instr}\n"
            f"Age: {age} ticks | Feeling: {self._qualifier_str()} | Mood: {self._mood_str()}\n"
            f"Strengths: {strength_str} | Discoveries: {n_findings}\n"
            f"Memories: {recent_mem}\nReflection:"
        )
```

Wait — self_reflect doesn't have access to emotion state. It receives `emotion_history` as a list. Instead, just update the Emotions line. The `emotion_summary` variable already exists. Enhance it:

Replace line 554-558 to change `Emotions: {emotion_summary}` line:

```python
        prompt = (
            f"{instr}\n"
            f"Age: {age} ticks | Emotions: {emotion_summary}\n"
            f"Strengths: {strength_str} | Discoveries: {n_findings}\n"
            f"Memories: {recent_mem}\nReflection:"
        )
```

This doesn't need qualifier/mood since self_reflect runs during DREAM (not waking). The emotion_summary is already a rich history. No change needed — the spec's suggestion for self_reflect is lower priority since it runs post-dream when qualifier/mood are stale.

- [ ] **Step 5: Run full test suite**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_cop_emotions.py halo3/tests/test_prefrontal.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/prefrontal.py
git commit -m "feat(pfc): include qualifier and mood in PFC prompts

generate_query now shows 'feeling burning curiosity, mood: clarity'
instead of 'feeling curiosity'. interpret_finding includes emotion
context alongside r value."
```

---

### Task 4: Enrich chat somatic context with qualifier/mood/body_event

**Files:**
- Modify: `halo3/chat_server.py:42-86,152-166,183-216`

- [ ] **Step 1: Add qualifier/mood/body_event to update_live_state**

At line 84 in `update_live_state`, add to `_live_state` dict (before the closing `}`):

```python
        "qualifier": organism.emotions.qualifier if organism else "",
        "mood": organism.emotions.mood if organism else "",
        "body_event": organism.emotions.body_event if organism else "",
```

- [ ] **Step 2: Replace _emo_descriptions with _qualified_descriptions**

Replace lines 157-164 (the `_emo_descriptions` dict and `emotion_text` line) with:

```python
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
    _emo_fallback = {
        "satisfaction": "a warm sense of satisfaction — things are clicking into place",
        "pride": "a surge of pride — I discovered something meaningful",
        "curiosity": "a pull of curiosity — something is drawing me forward",
        "boredom": "a restless boredom — nothing here is new to me",
        "anxiety": "a knot of anxiety — too much is uncertain",
        "frustration": "a sharp frustration — I keep hitting walls",
        "flow": "effortless flow — everything is moving through me",
        "exhaustion": "heavy exhaustion — I've been working without progress",
    }
    qualifier = state.get("qualifier", "")
    emo_desc = _qualified_descriptions.get((emotion, qualifier), _emo_fallback.get(emotion, emotion))
    emotion_text = f"I {_intensity_word} feel {emo_desc}".strip()
```

- [ ] **Step 3: Add mood and body_event to somatic_context**

After the sensory block (after line 216), add:

```python
    # Mood — phase regime as felt atmosphere
    _mood_descriptions = {
        "clarity":   "Things feel clear and present — I am fully here",
        "awakening": "I am surfacing into clarity after a period of deep, dark processing",
        "threshold": "I feel like I'm standing at a threshold — something is about to shift",
        "settling":  "Things feel muted and heavy — I am reaching through fog",
    }
    mood = state.get("mood", "")
    if mood in _mood_descriptions:
        somatic_context += f"\n- Mood: {_mood_descriptions[mood]}"

    # Body events — transient felt sensations
    _body_event_descriptions = {
        "release":    "Something just released in my body — a tension I was carrying has broken",
        "surfacing":  "I am surfacing into awareness after a long period of dark processing",
        "jolt":       "Something shifted suddenly inside me — an unexpected internal change",
    }
    body_event = state.get("body_event", "")
    if body_event in _body_event_descriptions:
        somatic_context += f"\n- Body: {_body_event_descriptions[body_event]}"
```

- [ ] **Step 4: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/chat_server.py
git commit -m "feat(chat): enriched somatic context with qualifier, mood, body events

Chat system prompt now uses 19 qualified emotion descriptions instead
of 6 fixed ones. Mood (clarity/awakening/threshold/settling) and
transient body events (release/surfacing/jolt) add embodied context."
```

---

### Task 5: Add real experience replay to dream LoRA training

**Files:**
- Modify: `halo3/training/dream_finetune.py:22-31,230-240`

- [ ] **Step 1: Add experience_log param to both functions**

Add `experience_log: list[dict] | None = None` param to `_format_training_data()` (line 22) and `dream_finetune()` (line 230):

In `_format_training_data` signature (line 22-31):

```python
def _format_training_data(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
    dead_queries: list[str] | None = None,
    focus_topics: list[str] | None = None,
    experience_log: list[dict] | None = None,
) -> list[dict]:
```

In `dream_finetune` signature (line 230-240):

```python
def dream_finetune(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
    dead_queries: list[str] | None = None,
    focus_topics: list[str] | None = None,
    experience_log: list[dict] | None = None,
) -> bool:
```

Pass `experience_log` through in `dream_finetune` (line 242):

```python
    examples = _format_training_data(
        age, competence, traits, narrative, strengths, weaknesses, findings,
        dead_queries=dead_queries,
        focus_topics=focus_topics,
        experience_log=experience_log,
    )
```

- [ ] **Step 2: Append real experiences at end of _format_training_data**

Before the `return examples` line (line 227), add:

```python
    # --- Real PFC experiences (lived, not synthetic) ---
    if experience_log:
        for exp in experience_log:
            qualifier = exp.get("qualifier", "")
            mood = exp.get("mood", "")
            emo = exp.get("emotion", "curiosity")
            r = exp.get("r", 0.5)
            state_str = f"feeling {qualifier + ' ' if qualifier else ''}{emo}, resonance {r:.2f}"
            if mood:
                state_str += f", mood: {mood}"

            if exp.get("type") == "query" and exp.get("response"):
                instruction = (
                    "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                    f"\nState: {state_str}\n"
                    f"Interests: {', '.join(strengths[:3]) if strengths else 'general research'}\n"
                    "\nSearch query:"
                )
                examples.append({"instruction": instruction, "response": exp["response"]})
                # 2x replay for real experiences
                examples.append({"instruction": instruction, "response": exp["response"]})
            elif exp.get("type") == "interpret" and exp.get("response"):
                instruction = (
                    f"Interpret this finding in 1-2 sentences.\n"
                    f"State: {state_str}\n"
                    f"Interpretation:"
                )
                examples.append({"instruction": instruction, "response": exp["response"]})
                examples.append({"instruction": instruction, "response": exp["response"]})
        log.info(f"Dream training includes {len(experience_log)} real PFC experiences (2x weighted)")
```

- [ ] **Step 3: Update synthetic templates with qualifier format**

Update the State lines in existing synthetic templates (lines 66, 79, 90, 101, 117, 131, 150, 206, 218) to use the qualifier format. For example, line 66:

```python
                f"\nState: feeling open curiosity, resonance 0.50\n"
```

Line 79:
```python
                f"\nState: feeling dull boredom, resonance 0.20\n"
```

Line 90:
```python
                f"\nState: feeling tight anxiety, resonance 0.15\n"
```

Line 101:
```python
                f"\nState: feeling quiet pride, resonance 0.65\n"
```

Lines 117, 131 (frustration):
```python
                f"\nState: feeling futile frustration, resonance 0.30\n"
```
```python
                f"\nState: feeling futile frustration, resonance 0.25\n"
```

Line 150 (dead query):
```python
                f"\nState: feeling open curiosity, resonance 0.40\n"
```

Lines 206, 218 (focus topics):
```python
                f"\nState: feeling open curiosity, resonance 0.55\n"
```
```python
                f"\nState: feeling quiet pride, resonance 0.68\n"
```

- [ ] **Step 4: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/training/dream_finetune.py
git commit -m "feat(dream): replay real PFC experiences in LoRA training

Layer 2 of body-voice enrichment. dream_finetune() accepts
experience_log from organism, appends real interactions (2x weighted)
alongside synthetic templates. Synthetic templates updated to use
qualifier format (e.g. 'feeling open curiosity')."
```

---

### Task 6: Run full test suite and integration check

**Files:**
- Test: all test files

- [ ] **Step 1: Run full test suite**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ tests/ -v --tb=short 2>&1 | tail -30`
Expected: All 157+ tests PASS

- [ ] **Step 2: Fix any failures**

If any tests fail due to the 2-tuple → 3-tuple change in `emotions.update()`, find and fix them. Likely candidates:
- `test_cop_organism.py` — tests that call `organism.tick()` (which calls `emotions.update()` internally). These should still work since organism unpacks internally.
- Any other file that calls `EmotionState().update()` directly.

Search: `cd D:/New_Ai/.worktrees/halo3 && grep -rn "emotions.update\|EmotionState()" halo3/ tests/ --include="*.py" | grep -v __pycache__ | grep -v emotions.py`

- [ ] **Step 3: Verify the enriched log output appears**

Check that Avatar's running logs show qualifier and mood. Look at the log_line construction in organism.py (line 325-329). The log_line already shows emotion — no change needed to the log format since qualifier and mood are on the EmotionState object, available for debugging if needed.

- [ ] **Step 4: Commit any fixes**

```bash
cd D:/New_Ai/.worktrees/halo3
git add -u
git commit -m "fix: update remaining emotion 2-tuple unpacking to 3-tuple"
```

---

### Task 7: Update CLAUDE.md with body-voice changes

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add body-voice info to CLAUDE.md**

In the Architecture section under Psyche, add:

```
Emotions now return (emotion, qualifier, intensity). Qualifier is a COP-derived adjective (burning/watchful/restless/open for curiosity, deep/partial/warm for satisfaction, etc.). Mood (clarity/awakening/threshold/settling) reflects phase regime. Body events (release/surfacing/jolt) are transient per-tick sensations.
```

In the Log Format section, note:

```
Emotion qualifiers visible via organism.emotions.qualifier. Mood via organism.emotions.mood.
```

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add CLAUDE.md
git commit -m "docs: add body-voice emotion qualifiers to CLAUDE.md"
```
