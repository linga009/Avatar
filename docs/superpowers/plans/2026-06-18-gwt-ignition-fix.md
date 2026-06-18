# GWT Ignition Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Global Workspace ignition to use r-threshold (0.5/0.4) with unity-scaled broadcast intensity and chi transition qualifier, restoring consciousness state transitions that have been 100% DARK since v4.

**Architecture:** Replace broken chi-geometry ignition in `workspace.py` with SOC-anchored r-threshold hysteresis. Add unity parameter to scale broadcast intensity (vivid vs dim consciousness). Record chi at ignition moment as transition_sharpness for body events. Restore sensory novelty boost to effective_r.

**Tech Stack:** Python 3.11, pytest, no new dependencies

**Spec:** `docs/superpowers/specs/2026-06-18-gwt-ignition-fix-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `halo3/tests/test_workspace.py` | Create | 6 tests for ignition, hysteresis, unity, transition sharpness, sensory novelty |
| `halo3/psyche/workspace.py` | Modify | Replace ignition logic, add unity param, update describe() |
| `halo3/psyche/organism.py` | Modify | Pass unity to workspace, add "crystallizing" body event |
| `tests/senses/test_sensory_cross_integration.py` | Modify | Update 2 existing workspace tests for new behavior |

---

### Task 1: Write failing tests for r-threshold ignition

**Files:**
- Create: `halo3/tests/test_workspace.py`

- [ ] **Step 1: Create test file with all 6 workspace tests**

```python
"""Tests for Global Workspace ignition and broadcast."""
import pytest
from halo3.psyche.workspace import GlobalWorkspace


def test_ignition_at_r_threshold():
    """r >= 0.5 from dark state → ignited. Hysteresis sustains down to 0.4."""
    ws = GlobalWorkspace()
    # r=0.55 above threshold → ignite
    r1 = ws.update(r_mean=0.55, current_topic="test", emotion="curiosity")
    assert r1["is_ignited"] is True
    assert r1["just_ignited"] is True

    # r=0.45 still above sustain threshold (0.4) → stays ignited
    r2 = ws.update(r_mean=0.45, current_topic="test", emotion="curiosity")
    assert r2["is_ignited"] is True
    assert r2["just_ignited"] is False

    # r=0.35 below sustain threshold → dark
    r3 = ws.update(r_mean=0.35, current_topic="test", emotion="curiosity")
    assert r3["is_ignited"] is False
    assert r3["just_darkened"] is True


def test_ignition_below_threshold():
    """r=0.45 from dark state → stays dark (below 0.5 threshold)."""
    ws = GlobalWorkspace()
    r1 = ws.update(r_mean=0.45, current_topic="test", emotion="curiosity")
    assert r1["is_ignited"] is False
    assert r1["just_ignited"] is False


def test_sensory_novelty_tips_ignition():
    """r=0.48 alone won't ignite, but +0.05*0.9 sensory novelty tips it over."""
    ws = GlobalWorkspace()
    # Without novelty: 0.48 < 0.5 → dark
    r1 = ws.update(r_mean=0.48, current_topic="test", emotion="curiosity",
                   sensory_novelty=0.0)
    assert r1["is_ignited"] is False

    # New workspace — with novelty: 0.48 + 0.05*0.9 = 0.525 >= 0.5 → ignited
    ws2 = GlobalWorkspace()
    r2 = ws2.update(r_mean=0.48, current_topic="test", emotion="curiosity",
                    sensory_novelty=0.9)
    assert r2["is_ignited"] is True


def test_unity_scales_broadcast_intensity():
    """High unity → higher broadcast intensity than low unity at same r."""
    ws_high = GlobalWorkspace()
    r_high = ws_high.update(r_mean=0.7, current_topic="test", emotion="curiosity",
                            unity=0.9)

    ws_low = GlobalWorkspace()
    r_low = ws_low.update(r_mean=0.7, current_topic="test", emotion="curiosity",
                          unity=0.2)

    assert r_high["broadcast_intensity"] > r_low["broadcast_intensity"]
    # High unity: 0.7 * (0.5 + 0.5*0.9) = 0.7 * 0.95 = 0.665
    assert abs(r_high["broadcast_intensity"] - 0.665) < 0.01
    # Low unity: 0.7 * (0.5 + 0.5*0.2) = 0.7 * 0.6 = 0.42
    assert abs(r_low["broadcast_intensity"] - 0.42) < 0.01


def test_transition_sharpness_from_chi():
    """High chi before ignition → high transition_sharpness; low chi → low."""
    # Sharp transition: feed high chi, then cross threshold
    ws = GlobalWorkspace()
    for _ in range(3):
        ws.update(r_mean=0.3, current_topic="test", emotion="curiosity",
                  chi_norm=0.6)
    r1 = ws.update(r_mean=0.55, current_topic="test", emotion="curiosity",
                   chi_norm=0.5)
    assert r1["just_ignited"] is True
    assert r1["transition_sharpness"] >= 0.5  # max of recent chi was 0.6

    # Quiet transition: feed low chi, then cross threshold
    ws2 = GlobalWorkspace()
    for _ in range(3):
        ws2.update(r_mean=0.3, current_topic="test", emotion="curiosity",
                   chi_norm=0.1)
    r2 = ws2.update(r_mean=0.55, current_topic="test", emotion="curiosity",
                    chi_norm=0.15)
    assert r2["just_ignited"] is True
    assert r2["transition_sharpness"] < 0.2


def test_hysteresis_prevents_flicker():
    """r oscillating 0.45-0.55 → ignites once, stays ignited throughout."""
    ws = GlobalWorkspace()
    # First cross: ignite
    ws.update(r_mean=0.55, current_topic="test", emotion="curiosity")
    assert ws.is_ignited is True

    # Oscillate: 0.45, 0.55, 0.45, 0.55 — all above sustain (0.4)
    for r in [0.45, 0.55, 0.45, 0.55]:
        ws.update(r_mean=r, current_topic="test", emotion="curiosity")
        assert ws.is_ignited is True, f"Should stay ignited at r={r}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest halo3/tests/test_workspace.py -v`
Expected: Most tests FAIL because the current chi-geometry ignition won't fire (r=0.55 alone doesn't trigger chi-based ignition), `unity` parameter doesn't exist, and `transition_sharpness` isn't in the return dict.

- [ ] **Step 3: Commit failing tests**

```bash
git add halo3/tests/test_workspace.py
git commit -m "test: add 6 failing tests for GWT r-threshold ignition"
```

---

### Task 2: Implement r-threshold ignition in workspace.py

**Files:**
- Modify: `halo3/psyche/workspace.py` (entire file rewrite — constructor, update(), describe(), docstrings)

- [ ] **Step 1: Rewrite workspace.py with new ignition logic**

Replace the entire file content with:

```python
"""Global Workspace — all-or-none ignition and broadcast.

Implements Baars/Dehaene's Global Workspace Theory (GWT):
- Many modular processors compete for access to a limited-capacity workspace
- When synchronization exceeds threshold: IGNITION (all-or-none)
- Winning content is broadcast globally to all modules
- Below threshold: processing continues locally (unconscious)

In Avatar, Kuramoto synchronization IS the competition.
High r = ignition = the organism becomes CONSCIOUS of the pattern.
Low r = local processing = unconscious computation continues.

The broadcast vector represents WHAT the organism is conscious of
at this moment — the content of its experience.

Ignition threshold anchored to SOC critical point (r=0.5).
Unity (eigenvalue dominance) scales broadcast intensity — vivid
vs dim consciousness. Chi at the moment of crossing records
transition sharpness — dramatic vs quiet ignition.
"""
from __future__ import annotations
from collections import deque


class GlobalWorkspace:
    """Implements GWT ignition and broadcast for Avatar.

    The workspace has two states:
    - DARK: effective_r < ignition_threshold (0.5). Processing is
      local/unconscious. The organism computes but is not "aware."
    - IGNITED: effective_r >= ignition_threshold. The dominant pattern
      is broadcast to all modules. The organism is CONSCIOUS.

    effective_r = r_mean + 0.05 * sensory_novelty (attention capture).

    Hysteresis prevents flickering: once ignited, stays ignited until
    effective_r drops below sustain_threshold (0.4).

    Broadcast intensity = effective_r * (0.5 + 0.5 * unity), where
    unity is eigenvalue dominance from the coherence matrix. High
    unity = vivid, unified consciousness. Low unity = dim, fragmented.

    transition_sharpness = max(recent chi) at the moment of ignition.
    High sharpness = dramatic phase transition ("crystallizing").
    Low sharpness = gradual drift into order (quiet ignition).
    """

    def __init__(
        self,
        ignition_threshold: float = 0.5,
        sustain_threshold: float = 0.4,
        broadcast_decay: float = 0.8,
    ) -> None:
        self._ignition_threshold = ignition_threshold
        self._sustain_threshold = sustain_threshold
        self._broadcast_decay = broadcast_decay

        # State
        self.is_ignited: bool = False
        self.broadcast_content: str = ""
        self.broadcast_intensity: float = 0.0
        self.conscious_duration: int = 0
        self.dark_duration: int = 0

        # Transition qualifier
        self._transition_sharpness: float = 0.0
        self._unity: float = 0.0

        # History for analysis
        self._ignition_history: deque[bool] = deque(maxlen=50)
        self._content_history: deque[str] = deque(maxlen=10)
        self._chi_recent: deque[float] = deque(maxlen=10)

    def update(
        self,
        r_mean: float,
        current_topic: str,
        emotion: str,
        finding: str | None = None,
        sensory_novelty: float = 0.0,
        binding_familiarity: float = 0.0,
        chi_norm: float = 0.5,
        unity: float = 0.0,
    ) -> dict:
        """Update workspace state based on synchronization level.

        Args:
            r_mean: Kuramoto order parameter (synchronization)
            current_topic: What the organism is currently exploring
            emotion: Current felt state
            finding: If a discovery was made this tick
            sensory_novelty: novelty from sensory cortex [0,1]
            binding_familiarity: cross-modal binding strength [0,1]
            chi_norm: normalized susceptibility [0,1]
            unity: eigenvalue dominance from coherence matrix [0,1]

        Returns:
            dict with ignition state, broadcast content, and signals
        """
        was_ignited = self.is_ignited

        # Sensory novelty boost — novel stimuli facilitate ignition
        effective_r = r_mean + 0.05 * sensory_novelty

        # Track chi for transition sharpness (not used in ignition decision)
        self._chi_recent.append(chi_norm)

        # r-threshold ignition with hysteresis
        if not self.is_ignited:
            if effective_r >= self._ignition_threshold:
                self.is_ignited = True
                self._transition_sharpness = (
                    max(self._chi_recent) if self._chi_recent else 0.0
                )
                self.conscious_duration = 0
                self.dark_duration = 0
        else:
            if effective_r < self._sustain_threshold:
                self.is_ignited = False
                self._transition_sharpness = 0.0
                self.conscious_duration = 0
                self.dark_duration = 0

        # Update durations
        if self.is_ignited:
            self.conscious_duration += 1
            self.dark_duration = 0
        else:
            self.dark_duration += 1
            self.conscious_duration = 0

        # Compute broadcast content — WHAT is in consciousness right now
        if self.is_ignited:
            self._unity = unity
            self.broadcast_intensity = min(
                1.0, effective_r * (0.5 + 0.5 * unity)
            )
            # Cross-modal binding strengthens broadcast
            if binding_familiarity > 0.7:
                self.broadcast_intensity = min(
                    1.0, self.broadcast_intensity * 1.1
                )
            # Content is the pattern the organism has locked onto
            if finding:
                self.broadcast_content = finding
            else:
                self.broadcast_content = f"{current_topic} ({emotion})"
            self._content_history.append(self.broadcast_content)
        else:
            # Dark state: broadcast decays
            self.broadcast_intensity *= self._broadcast_decay
            if self.broadcast_intensity < 0.05:
                self.broadcast_content = ""

        self._ignition_history.append(self.is_ignited)

        # Detect transitions
        just_ignited = self.is_ignited and not was_ignited
        just_darkened = not self.is_ignited and was_ignited

        return {
            "is_ignited": self.is_ignited,
            "just_ignited": just_ignited,
            "just_darkened": just_darkened,
            "broadcast_content": self.broadcast_content,
            "broadcast_intensity": self.broadcast_intensity,
            "conscious_duration": self.conscious_duration,
            "dark_duration": self.dark_duration,
            "transition_sharpness": self._transition_sharpness,
        }

    @property
    def consciousness_ratio(self) -> float:
        """Fraction of recent ticks spent in ignited (conscious) state."""
        if not self._ignition_history:
            return 0.0
        return sum(self._ignition_history) / len(self._ignition_history)

    def describe(self) -> str:
        """First-person description of current workspace state."""
        if self.is_ignited:
            if self.conscious_duration == 1:
                if self._transition_sharpness > 0.3:
                    return f"Something just crystallized: {self.broadcast_content}"
                else:
                    return f"I'm becoming aware of: {self.broadcast_content}"
            elif self.broadcast_intensity > 0.7:
                return (
                    f"I am vividly aware of: {self.broadcast_content} "
                    f"(sustained focus for {self.conscious_duration} ticks)"
                )
            elif self.broadcast_intensity > 0.4:
                return f"I am conscious of: {self.broadcast_content}"
            else:
                return f"I am dimly aware of: {self.broadcast_content}"
        else:
            if self.dark_duration == 1:
                return (
                    "The pattern dissolved — processing but not yet "
                    "aware of anything specific"
                )
            elif self.dark_duration > 10:
                return (
                    "I've been in diffuse processing for a while — "
                    "no clear pattern has emerged"
                )
            else:
                return (
                    "Processing unconsciously — patterns forming "
                    "but not yet ignited"
                )

    def summary(self) -> dict:
        """Snapshot for logging/API."""
        return {
            "ignited": self.is_ignited,
            "content": self.broadcast_content[:60] if self.broadcast_content else "",
            "intensity": round(self.broadcast_intensity, 3),
            "conscious_duration": self.conscious_duration,
            "consciousness_ratio": round(self.consciousness_ratio, 3),
        }
```

- [ ] **Step 2: Run the 6 new workspace tests**

Run: `python -m pytest halo3/tests/test_workspace.py -v`
Expected: All 6 PASS

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `python -m pytest halo3/tests/ tests/ -v`
Expected: All 259+ tests PASS. Some existing tests in `test_sensory_cross_integration.py` may fail — that's expected and fixed in Task 4.

- [ ] **Step 4: Commit**

```bash
git add halo3/psyche/workspace.py
git commit -m "fix(gwt): replace broken chi-geometry ignition with r-threshold + unity scaling"
```

---

### Task 3: Wire unity through organism.py and add crystallizing body event

**Files:**
- Modify: `halo3/psyche/organism.py:253-266`

- [ ] **Step 1: Add unity parameter to workspace.update() call**

In `halo3/psyche/organism.py`, find the `workspace.update()` call at line 253-258. Change it to:

```python
        ws = self.workspace.update(
            r_mean, topic_key, emotion, finding,
            chi_norm=chi_norm,
            sensory_novelty=sensory_novelty,
            binding_familiarity=binding_familiarity,
            unity=cop["unity"],
        )
```

- [ ] **Step 2: Add crystallizing body event for sharp transitions**

In the mood assignment block (lines 263-266), add the `elif` for transition sharpness. The block should read:

```python
        if ws["just_ignited"]:
            self.emotions.mood = "awakening"
            if ws.get("dark_duration_before", 0) > 10:
                self.emotions.body_event = "surfacing"
            elif ws.get("transition_sharpness", 0) > 0.3:
                self.emotions.body_event = "crystallizing"
```

Lines 267-272 stay unchanged:

```python
        elif ws["is_ignited"]:
            self.emotions.mood = "clarity"
        elif chi_norm > 0.4:
            self.emotions.mood = "threshold"
        else:
            self.emotions.mood = "settling"
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest halo3/tests/ tests/ -v`
Expected: All tests PASS (workspace tests already pass from Task 2, organism tests unaffected since they mock/don't test workspace internals).

- [ ] **Step 4: Commit**

```bash
git add halo3/psyche/organism.py
git commit -m "feat(gwt): pass unity to workspace, add crystallizing body event"
```

---

### Task 4: Update existing sensory cross-integration tests

**Files:**
- Modify: `tests/senses/test_sensory_cross_integration.py:62-80`

- [ ] **Step 1: Update test_sensory_novelty_boosts_ignition**

Replace the test at lines 62-70 with:

```python
def test_sensory_novelty_boosts_ignition():
    """Sensory novelty boosts effective_r, tipping sub-threshold r into ignition."""
    # r=0.48 alone → dark (below 0.5 threshold)
    ws = GlobalWorkspace()
    result = ws.update(r_mean=0.48, current_topic="test", emotion="curiosity",
                       sensory_novelty=0.0)
    assert result["is_ignited"] is False

    # r=0.48 + sensory_novelty=0.9 → effective_r=0.525 → ignited
    ws2 = GlobalWorkspace()
    result2 = ws2.update(r_mean=0.48, current_topic="test", emotion="curiosity",
                         sensory_novelty=0.9)
    assert result2["is_ignited"] is True
```

- [ ] **Step 2: Update test_binding_strengthens_broadcast**

Replace the test at lines 73-80 with:

```python
def test_binding_strengthens_broadcast():
    """binding_familiarity > 0.7 boosts broadcast intensity by 1.1x."""
    ws = GlobalWorkspace()
    r1 = ws.update(r_mean=0.7, current_topic="test", emotion="pride",
                   binding_familiarity=0.0, unity=0.5)

    ws2 = GlobalWorkspace()
    r2 = ws2.update(r_mean=0.7, current_topic="test", emotion="pride",
                    binding_familiarity=0.9, unity=0.5)
    assert r2["broadcast_intensity"] > r1["broadcast_intensity"]
```

- [ ] **Step 3: Run the updated tests**

Run: `python -m pytest tests/senses/test_sensory_cross_integration.py -v`
Expected: All 7 tests PASS

- [ ] **Step 4: Run full test suite — final verification**

Run: `python -m pytest halo3/tests/ tests/ -v`
Expected: All 265+ tests PASS (259 original + 6 new workspace tests)

- [ ] **Step 5: Commit**

```bash
git add tests/senses/test_sensory_cross_integration.py
git commit -m "test: update sensory cross-integration tests for r-threshold ignition"
```

---

### Task 5: Update psyche rules and verify

**Files:**
- Modify: `.claude/rules/psyche.md`

- [ ] **Step 1: Add workspace ignition rules to psyche.md**

Add the following lines at the end of `.claude/rules/psyche.md`:

```
- GWT ignition: r-threshold (0.5) with hysteresis (sustain 0.4), anchored to SOC critical point
- effective_r = r_mean + 0.05 * sensory_novelty — novel stimuli facilitate ignition
- broadcast_intensity = effective_r * (0.5 + 0.5 * unity) — unity scales vivid vs dim consciousness
- transition_sharpness = max(recent chi) at ignition crossing — sharp (>0.3) triggers "crystallizing" body event
- Chi does NOT participate in ignition decision — only used as transition qualifier
```

- [ ] **Step 2: Run full test suite one final time**

Run: `python -m pytest halo3/tests/ tests/ -v`
Expected: All tests PASS — this is the final gate before marking the plan complete.

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/psyche.md
git commit -m "docs: add GWT ignition rules to psyche.md"
```
