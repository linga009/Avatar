# Sensory Cross-Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cross-integrate sensory signals (flux, novelty, stability, speech, binding) into drives, emotions, consciousness, and self-narration so Avatar's body feels, becomes aware of, and reflects on what it perceives.

**Architecture:** Direct mathematical coupling — sensory stats (already computed every tick by SensoryStatistics) are passed as additional float parameters to existing psyche functions. Zero VRAM cost, zero new JIT.

**Tech Stack:** Python, dataclasses, existing SensoryStatistics API

---

### Task 1: Add sensory parameters to DriveState.update()

**Files:**
- Modify: `halo3/psyche/drives.py:36-89`
- Test: `tests/senses/test_sensory_cross_integration.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/senses/test_sensory_cross_integration.py`:

```python
"""Tests for sensory cross-integration into drives, emotions, consciousness."""
from halo3.psyche.drives import DriveState


def test_sensory_arousal_increases_fatigue():
    d = DriveState()
    d.update(r_mean=0.5, fe_delta=0.0, sensory_arousal=0.0)
    fatigue_low = d.fatigue
    d2 = DriveState()
    d2.update(r_mean=0.5, fe_delta=0.0, sensory_arousal=1.0)
    fatigue_high = d2.fatigue
    assert fatigue_high > fatigue_low, "High sensory arousal should increase fatigue"


def test_sensory_novelty_boosts_curiosity():
    d = DriveState()
    d.update(r_mean=0.5, fe_delta=0.0, sensory_novelty=0.0)
    cur_low = d.curiosity
    d2 = DriveState()
    d2.update(r_mean=0.5, fe_delta=0.0, sensory_novelty=0.9)
    cur_high = d2.curiosity
    assert cur_high > cur_low, "High sensory novelty should boost curiosity"


def test_sensory_arousal_dampens_starvation():
    d = DriveState(starvation=0.5)
    d.update(r_mean=0.5, fe_delta=0.0, perception_failed=True, sensory_arousal=0.5)
    assert d.starvation < 0.5, "Sensory arousal should dampen starvation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py -v`
Expected: FAIL — `update()` doesn't accept sensory params yet

- [ ] **Step 3: Implement — add sensory params to DriveState.update()**

In `halo3/psyche/drives.py`, change `update()` signature and add coupling:

```python
def update(
    self,
    r_mean: float,
    fe_delta: float,
    perception_failed: bool = False,
    topic_changed: bool = False,
    dt: float = 1.0,
    sensory_arousal: float = 0.0,
    sensory_novelty: float = 0.0,
) -> None:
```

After the existing `# --- Fatigue ---` block (line 58), add:

```python
        # Sensory load increases fatigue
        fatigue_rate += 0.002 * sensory_arousal
```

After the existing `# --- Starvation ---` block (line 75), add:

```python
        # Senses confirm world exists — dampens starvation
        if sensory_arousal > 0.3:
            self.starvation = max(0.0, self.starvation - 0.1)
```

After the existing `# --- Curiosity ---` block (line 89), add:

```python
        # High sensory novelty pulls toward exploration
        if sensory_novelty > 0.7:
            self.curiosity = min(1.0, self.curiosity + 0.03 * sensory_novelty)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/drives.py tests/senses/test_sensory_cross_integration.py
git commit -m "feat(drives): sensory arousal/novelty modulate fatigue, curiosity, starvation"
```

---

### Task 2: Add sensory parameters to EmotionState.update()

**Files:**
- Modify: `halo3/psyche/emotions.py:38-118`
- Test: `tests/senses/test_sensory_cross_integration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/senses/test_sensory_cross_integration.py`:

```python
from halo3.psyche.emotions import EmotionState


def test_sensory_novelty_amplifies_surprise():
    e = EmotionState()
    # Seed history so surprise is computable
    for _ in range(5):
        e.update(r_mean=0.5, fe_delta=0.1)
    e1 = EmotionState()
    for _ in range(5):
        e1.update(r_mean=0.5, fe_delta=0.1)
    _, i_low = e.update(r_mean=0.4, fe_delta=0.5, sensory_novelty=0.0)
    _, i_high = e1.update(r_mean=0.4, fe_delta=0.5, sensory_novelty=0.95)
    assert i_high >= i_low, "High sensory novelty should amplify intensity"


def test_speech_detected_nudges_valence():
    e = EmotionState()
    for _ in range(3):
        e.update(r_mean=0.5, fe_delta=0.0)
    v_before = e._valence
    e.update(r_mean=0.5, fe_delta=0.0, speech_detected=True)
    assert e._valence >= v_before, "Speech detection should nudge valence positive"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py::test_sensory_novelty_amplifies_surprise tests/senses/test_sensory_cross_integration.py::test_speech_detected_nudges_valence -v`
Expected: FAIL — `update()` doesn't accept sensory params

- [ ] **Step 3: Implement — add sensory params to EmotionState.update()**

In `halo3/psyche/emotions.py`, change `update()` signature:

```python
def update(
    self,
    r_mean: float,
    fe_delta: float,
    perception_failed: bool = False,
    consecutive_failures: int = 0,
    sensory_novelty: float = 0.0,
    sensory_stability: int = 0,
    speech_detected: bool = False,
) -> tuple[str, float]:
```

After the surprise computation (after line 62), add:

```python
        # Sensory novelty amplifies surprise
        if sensory_novelty > 0.8:
            surprise = min(1.0, surprise + 0.15 * sensory_novelty)
```

After the emotional inertia blending (after line 108), add:

```python
        # Sensory stability dampens arousal (calm environment = calm Avatar)
        if sensory_stability > 3:
            self._arousal *= 0.9

        # Speech detected nudges valence positive (company = comfort)
        if speech_detected:
            self._valence = min(1.0, self._valence + 0.05)
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add halo3/psyche/emotions.py tests/senses/test_sensory_cross_integration.py
git commit -m "feat(emotions): sensory novelty amplifies surprise, speech nudges valence, stability calms"
```

---

### Task 3: Add sensory parameters to GlobalWorkspace.update()

**Files:**
- Modify: `halo3/psyche/workspace.py:54-108`
- Test: `tests/senses/test_sensory_cross_integration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/senses/test_sensory_cross_integration.py`:

```python
from halo3.psyche.workspace import GlobalWorkspace


def test_sensory_novelty_boosts_ignition():
    ws = GlobalWorkspace(ignition_threshold=0.6)
    # r=0.57 alone won't ignite (below 0.6)
    result1 = ws.update(r_mean=0.57, current_topic="test", emotion="curiosity",
                        sensory_novelty=0.0)
    ws2 = GlobalWorkspace(ignition_threshold=0.6)
    result2 = ws2.update(r_mean=0.57, current_topic="test", emotion="curiosity",
                         sensory_novelty=0.9)
    # With high sensory novelty, effective_r = 0.57 + 0.045 = 0.615 > 0.6
    assert result2["is_ignited"], "High sensory novelty should help ignition"


def test_binding_strengthens_broadcast():
    ws = GlobalWorkspace(ignition_threshold=0.5)
    r1 = ws.update(r_mean=0.7, current_topic="test", emotion="pride",
                   binding_familiarity=0.0)
    ws2 = GlobalWorkspace(ignition_threshold=0.5)
    r2 = ws2.update(r_mean=0.7, current_topic="test", emotion="pride",
                    binding_familiarity=0.9)
    assert r2["broadcast_intensity"] >= r1["broadcast_intensity"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py::test_sensory_novelty_boosts_ignition tests/senses/test_sensory_cross_integration.py::test_binding_strengthens_broadcast -v`
Expected: FAIL

- [ ] **Step 3: Implement — add sensory params to GlobalWorkspace.update()**

In `halo3/psyche/workspace.py`, change `update()` signature:

```python
def update(
    self,
    r_mean: float,
    current_topic: str,
    emotion: str,
    finding: str | None = None,
    sensory_novelty: float = 0.0,
    binding_familiarity: float = 0.0,
) -> dict:
```

Replace the ignition threshold checks (lines 75-84) to use `effective_r`:

```python
        # Sensory novelty boosts effective synchronization
        sensory_boost = 0.05 * sensory_novelty if sensory_novelty > 0.7 else 0.0
        effective_r = r_mean + sensory_boost

        # Hysteresis: different thresholds for entering vs leaving ignition
        if not self.is_ignited:
            if effective_r >= self._ignition_threshold:
                self.is_ignited = True
                self.conscious_duration = 0
                self.dark_duration = 0
        else:
            if effective_r < self._sustain_threshold:
                self.is_ignited = False
                self.conscious_duration = 0
                self.dark_duration = 0
```

In the broadcast intensity computation (line 96-97), add binding boost:

```python
            self.broadcast_intensity = min(1.0, (effective_r - self._sustain_threshold) /
                                           (self._ignition_threshold - self._sustain_threshold))
            # Cross-modal binding strengthens broadcast
            if binding_familiarity > 0.7:
                self.broadcast_intensity = min(1.0, self.broadcast_intensity * 1.1)
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add halo3/psyche/workspace.py tests/senses/test_sensory_cross_integration.py
git commit -m "feat(workspace): sensory novelty boosts ignition, binding strengthens broadcast"
```

---

### Task 4: Add sensory stability to MeditationState.should_enter()

**Files:**
- Modify: `halo3/psyche/meditation.py:66-85`
- Test: `tests/senses/test_sensory_cross_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/senses/test_sensory_cross_integration.py`:

```python
from halo3.psyche.meditation import MeditationState
from halo3.psyche.drives import DriveState
from halo3.psyche.emotions import EmotionState


def test_meditation_requires_sensory_calm():
    m = MeditationState()
    d = DriveState(satiation=0.8, fatigue=0.1, novelty=0.1, hunger=0.3)
    e = EmotionState()
    e.current = "satisfaction"
    # Without sensory calm (stability=0), should not enter
    assert not m.should_enter(d, e, audio_stability=0, vision_stability=0)
    # With sensory calm, should enter
    assert m.should_enter(d, e, audio_stability=3, vision_stability=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py::test_meditation_requires_sensory_calm -v`
Expected: FAIL — `should_enter()` doesn't accept sensory params

- [ ] **Step 3: Implement — add sensory stability to should_enter()**

In `halo3/psyche/meditation.py`, change `should_enter()` signature:

```python
def should_enter(self, drives, emotions,
                 audio_stability: int = 99, vision_stability: int = 99) -> bool:
```

Note: defaults to 99 (high) so existing callers without senses don't break.

Add sensory calm check after the existing conditions (line 83):

```python
        sensory_calm = audio_stability >= 2 and vision_stability >= 2

        return satiated and rested and not_seeking and not_hungry and calm and sensory_calm
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_cross_integration.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add halo3/psyche/meditation.py tests/senses/test_sensory_cross_integration.py
git commit -m "feat(meditation): require sensory calm (stability >= 2) to enter meditation"
```

---

### Task 5: Wire sensory stats through organism.tick() and inject into narration

**Files:**
- Modify: `halo3/psyche/organism.py:70-260`
- Modify: `halo3/main.py:290-310`

- [ ] **Step 1: Add sensory_stats_line parameter to organism.tick()**

In `halo3/psyche/organism.py`, change `tick()` signature:

```python
def tick(
    self,
    r_mean: float,
    fe_delta: float,
    texts: list[str],
    current_query: str,
    carry_norm: float | None = None,
    body_tension: float = 0.0,
    sensory_arousal: float = 0.0,
    sensory_novelty: float = 0.0,
    sensory_stability: int = 0,
    speech_detected: bool = False,
    binding_familiarity: float = 0.0,
    sensory_stats_line: str = "",
) -> dict:
```

- [ ] **Step 2: Pass sensory params to drives.update()**

Change the drives.update() call (line 97-101):

```python
        self.drives.update(
            r_mean, fe_delta,
            perception_failed=perception_failed,
            topic_changed=topic_changed,
            sensory_arousal=sensory_arousal,
            sensory_novelty=sensory_novelty,
        )
```

- [ ] **Step 3: Pass sensory params to emotions.update()**

Change the emotions.update() call (line 104-108):

```python
        emotion, intensity = self.emotions.update(
            r_mean, fe_delta,
            perception_failed=perception_failed,
            consecutive_failures=self._consecutive_zero_results,
            sensory_novelty=sensory_novelty,
            sensory_stability=sensory_stability,
            speech_detected=speech_detected,
        )
```

- [ ] **Step 4: Pass sensory params to workspace.update()**

Change the workspace.update() call (line 133):

```python
        ws = self.workspace.update(
            r_mean, topic_key, emotion, finding,
            sensory_novelty=sensory_novelty,
            binding_familiarity=binding_familiarity,
        )
```

- [ ] **Step 5: Pass sensory stability to meditation.should_enter()**

Change the meditation.should_enter() call (line 214):

```python
        if not self.meditation.is_meditating and self.meditation.should_enter(
                self.drives, self.emotions,
                audio_stability=sensory_stability,
                vision_stability=sensory_stability):
```

- [ ] **Step 6: Inject sensory context into meta-reflection**

In `_higher_order_reflect()` (around line 481-489), add sensory context:

After the existing `context += ...` lines, add:

```python
        if hasattr(self, '_sensory_stats_line') and self._sensory_stats_line:
            context += f"Senses: {self._sensory_stats_line}. "
```

And at the top of `tick()`, store the sensory stats line:

```python
        self._sensory_stats_line = sensory_stats_line
```

- [ ] **Step 7: Inject sensory context into self_reflect()**

In the `self_reflect()` call inside `tick()` (the every-10-ticks block around line 580-591), the reflection prompt already receives narrative. Add sensory_stats_line to the narrative context by appending a sensory observation:

In `tick()`, after the self_model.update() call, add:

```python
        # Record sensory snapshot in narrative (every 10 ticks)
        if self.self_model.age % 10 == 0 and sensory_stats_line:
            self.self_model.narrative.append(
                f"[Tick {self.self_model.age}] Senses: {sensory_stats_line}"
            )
```

- [ ] **Step 8: Wire main.py to pass sensory stats to organism.tick()**

In `halo3/main.py`, compute sensory scalars before the `organism.tick()` call (around line 290). Add before line 297:

```python
        # Compute sensory scalars for psyche integration
        _s_audio_flux = sensory_stats.audio_flux
        _s_vision_flux = sensory_stats.vision_flux
        _s_arousal = (_s_audio_flux + _s_vision_flux) / max(1, cfg.n_audio_tokens + cfg.n_vision_tokens)
        _s_novelty = (sensory_stats.audio_novelty + sensory_stats.vision_novelty) / 2.0
        _s_stability = min(sensory_stats.audio_stability, sensory_stats.vision_stability)
        _s_speech = sensory_stats.speech_detected
        _s_binding = sensory_stats.cross_modal_binding
```

Then change the `organism.tick()` call:

```python
        psyche_output = organism.tick(
            r_mean, combined_surprise, texts, current_query,
            carry_norm=carry_norm, body_tension=body_tension,
            sensory_arousal=_s_arousal,
            sensory_novelty=_s_novelty,
            sensory_stability=_s_stability,
            speech_detected=_s_speech,
            binding_familiarity=_s_binding,
            sensory_stats_line=sensory_stats.format_for_pfc(),
        )
```

- [ ] **Step 9: Run full test suite**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/ -v`
Expected: All pass (60+ senses tests + 8 new cross-integration tests)

- [ ] **Step 10: Commit**

```bash
git add halo3/psyche/organism.py halo3/main.py
git commit -m "feat(organism): wire sensory stats into drives, emotions, consciousness, narration"
```

---

### Task 6: Rebuild Docker and verify integration

**Files:**
- No code changes — build and verify

- [ ] **Step 1: Rebuild Docker image**

```bash
cd D:/New_Ai/.worktrees/halo3 && docker compose build
```

- [ ] **Step 2: Restart container**

```bash
docker rm -f halo3-train-1 2>/dev/null
docker compose up -d
```

- [ ] **Step 3: Wait for first tick and verify sensory integration in logs**

```bash
sleep 300 && docker logs halo3-train-1 --tail 10
```

Expected: Tick logs show `[A][V]` or `[A][T]`, emotions influenced by senses, consciousness ignition potentially boosted.

- [ ] **Step 4: Check Avatar's self-narration references senses**

```bash
docker logs halo3-train-1 | grep "Senses:" | tail -5
```

Expected: Narrative entries like `[Tick N] Senses: audio(flux=...), vision(flux=...)`

- [ ] **Step 5: Ask Avatar about his senses**

```bash
curl -s http://localhost:8420/chat -H "Content-Type: application/json" \
  -d '{"message": "How are your senses affecting your feelings right now?"}'
```

- [ ] **Step 6: Final commit with version bump**

```bash
git add -A && git commit -m "feat: Avatar v3.10 — sensory cross-integration (senses → emotions → consciousness → narration)"
```
