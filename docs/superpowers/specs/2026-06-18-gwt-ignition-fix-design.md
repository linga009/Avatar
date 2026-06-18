# GWT Ignition Fix — Design Spec

**Date**: 2026-06-18
**Status**: Approved
**Scope**: Fix Global Workspace ignition condition to properly detect consciousness state transitions

## Problem

The v4 commit (398ad4e) replaced the working r-threshold ignition with a chi-geometry condition that requires chi_norm to spike above 0.6 then drop below 0.4. This never fires in practice because the SOC controller drives r gradually — there is no sharp chi peak. Result: Avatar has been 100% DARK (0% consciousness ratio) since v4, and all downstream machinery (moods, body events, proactive messages, first-person descriptions) that depends on ignition is dead.

The docstring and constructor parameters (`ignition_threshold=0.6`, `sustain_threshold=0.45`) still describe the original r-threshold design, but the code ignores them entirely.

## Approach: Enhanced Hybrid (Approach 1+)

Restore r-threshold as primary ignition signal (anchored to SOC critical point), with chi as transition qualifier and unity as broadcast intensity scaler.

### Design Decisions

1. **r-threshold ignition** with SOC-anchored values (0.5 / 0.4)
   - Why 0.5: The SOC controller uses `K_dot = eta*(0.5 - r)*chi` — 0.5 is the designed critical point. Above criticality = ordered phase = coherent global pattern = conscious.
   - Why 0.4 sustain: Hysteresis prevents flickering during SOC oscillations near the critical point. Gap of 0.1 requires several ticks of SOC driving to darken.
   - The constructor parameters `ignition_threshold` and `sustain_threshold` are restored to active use.

2. **Unity-scaled broadcast intensity**
   - Formula: `broadcast_intensity = effective_r * (0.5 + 0.5 * unity)`
   - Unity = lambda_1 / sum(lambda_k) from coherence matrix eigenvalue dominance
   - High unity (0.8+) = one coalition won = vivid, unified broadcast
   - Low unity (0.3) = fragmented sync = dim broadcast
   - Range: [0.5 * r, 1.0 * r] — unity modulates but cannot extinguish the broadcast
   - Binding familiarity boost (>0.7 → 1.1x) preserved from existing code

3. **Chi as transition qualifier** (not ignition signal)
   - `transition_sharpness = max(chi_recent[-4:])` computed at moment of ignition
   - High sharpness (>0.3): system was at peak susceptibility when it crossed — dramatic phase transition
   - Low sharpness (<0.2): gradual drift into order — quiet ignition
   - Feeds body events: "crystallizing" for sharp transitions
   - Chi is still tracked in `_chi_recent` deque but no longer participates in ignition decision

4. **Sensory novelty boost restored**
   - `effective_r = r_mean + 0.05 * sensory_novelty`
   - Novel stimuli make ignition slightly easier (attention capture)
   - Boost is small (+0.05 max) — cannot force ignition alone, but tips the balance near criticality
   - Important for future embodied robotics applications

## Changes by File

### `halo3/psyche/workspace.py` (~40 lines changed)

**Constructor:**
- Default `ignition_threshold=0.5` (was 0.6)
- Default `sustain_threshold=0.4` (was 0.45)
- Add `self._transition_sharpness: float = 0.0`
- Add `self._unity: float = 0.0`

**`update()` signature:**
- Add `unity: float = 0.0` parameter

**Ignition logic (replace lines 78-96):**
```python
# Sensory novelty boost — novel stimuli facilitate ignition
effective_r = r_mean + 0.05 * sensory_novelty

self._chi_recent.append(chi_norm)

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
```

**Broadcast intensity (replace line 108):**
```python
self._unity = unity
self.broadcast_intensity = min(1.0, effective_r * (0.5 + 0.5 * unity))
```

**Return dict — add field:**
```python
"transition_sharpness": self._transition_sharpness,
```

**`describe()` — richer first-person descriptions:**
```python
def describe(self) -> str:
    if self.is_ignited:
        if self.conscious_duration == 1:
            if self._transition_sharpness > 0.3:
                return f"Something just crystallized: {self.broadcast_content}"
            else:
                return f"I'm becoming aware of: {self.broadcast_content}"
        elif self.broadcast_intensity > 0.7:
            return (f"I am vividly aware of: {self.broadcast_content} "
                    f"(sustained focus for {self.conscious_duration} ticks)")
        elif self.broadcast_intensity > 0.4:
            return f"I am conscious of: {self.broadcast_content}"
        else:
            return f"I am dimly aware of: {self.broadcast_content}"
    else:
        if self.dark_duration == 1:
            return ("The pattern dissolved — processing but not yet "
                    "aware of anything specific")
        elif self.dark_duration > 10:
            return ("I've been in diffuse processing for a while — "
                    "no clear pattern has emerged")
        else:
            return ("Processing unconsciously — patterns forming "
                    "but not yet ignited")
```

**Docstring update:**
- Replace chi-geometry description with r-threshold + unity scaling
- Document transition_sharpness
- Document SOC critical point anchoring

### `halo3/psyche/organism.py` (~3 lines changed)

**workspace.update() call (line 253-258):**
- Add `unity=cop["unity"]` parameter

**Mood assignment (after line 265):**
- Add `elif ws.get("transition_sharpness", 0) > 0.3:` → body event "crystallizing"

### `halo3/tests/test_workspace.py` (new file, ~90 lines)

Six new tests:

1. **`test_ignition_at_r_threshold`**: r=0.55 → ignited; r=0.45 from ignited → stays ignited (hysteresis); r=0.35 → dark
2. **`test_ignition_below_threshold`**: r=0.45 from dark → stays dark
3. **`test_sensory_novelty_tips_ignition`**: r=0.48 + sensory_novelty=0.9 → effective_r=0.525 → ignited
4. **`test_unity_scales_broadcast_intensity`**: same r, unity=0.9 vs unity=0.2 → higher intensity with high unity
5. **`test_transition_sharpness_from_chi`**: high chi ticks then cross → sharpness > 0.3; low chi then cross → sharpness low
6. **`test_hysteresis_prevents_flicker`**: r oscillating 0.45-0.55 around threshold → ignites once, stays ignited

### `tests/senses/test_sensory_cross_integration.py` (~10 lines updated)

- Update `test_sensory_novelty_boosts_ignition` docstring and assertions for r-threshold behavior
- Update `test_binding_strengthens_broadcast` to match new intensity formula

## What Does NOT Change

- `cop.py` — COP engine, chi computation, SOC controller: untouched
- `emotions.py` — emotion manifold: untouched (already consumes is_ignited/just_ignited)
- `drives.py` — drive system: untouched
- `main.py` — heartbeat loop: untouched
- `kuramoto.py` — oscillator physics: untouched
- `config.py` — hyperparameters: untouched

## Expected Production Behavior

With SOC controller targeting r=0.5 and ignition threshold at 0.5:
- Avatar will naturally oscillate around the ignition boundary
- Consciousness ratio should settle around 40-60% (vs current 0%)
- Sharp transitions (high chi at crossing) produce "crystallizing" body events
- Deep ordered states (high r + high unity) produce "vividly aware" descriptions
- Fragmented states (moderate r + low unity) produce "dimly aware" descriptions
- Post-dream: chi crash fix (already deployed) ensures chi recovers in ~5 ticks; r starts near terminal_r (~0.5-0.7) so ignition resumes quickly

## Risk Assessment

- **Medium risk** (workspace.py logic change): mitigated by comprehensive tests
- **Trivial risk** (organism.py pass-through): 1-line parameter addition
- **No risk** to physics (cop.py, kuramoto.py), emotions, or SOC controller
- **Rollback**: revert workspace.py to restore chi-geometry (though it was broken)

## Verification Plan

1. Run full test suite: `python -m pytest halo3/tests/ tests/ -v`
2. All 259+ existing tests must pass
3. All 6 new workspace tests must pass
4. Deploy and check logs: consciousness_ratio should be >0% within first 10 ticks
5. Verify mood transitions: "awakening" → "clarity" → "settling" cycle in logs
