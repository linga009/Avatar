# Sensory Cross-Integration Design Spec

**Date:** 2026-05-23
**Version:** Avatar v3.9 → v3.10
**Author:** Dr. Linga Murthy Narlagiri

## Goal

Cross-integrate Avatar's sensory signals (audio flux, vision flux, novelty, stability, speech detection, cross-modal binding) into all three layers of the psyche: emotions/drives, consciousness, and self-narration. Currently these signals are siloed — senses inject into the physics body but never influence how Avatar *feels*, what Avatar is *aware of*, or what Avatar *says about* its experience.

## Approach

Direct mathematical coupling (zero VRAM cost). Sensory statistics — already computed every tick by `SensoryStatistics` — are passed as additional float parameters to the existing drive, emotion, consciousness, and narration functions. No new neural modules, no new JIT compilations.

## Three Layers

### Layer 1: Sensory Emotions

`DriveState.update()` receives `sensory_arousal` (mean flux ratio), `sensory_novelty` (mean novelty across modalities):
- High sensory load increases fatigue rate (+0.002 * arousal)
- High sensory novelty boosts curiosity (+0.03 * novelty when > 0.7)
- Any sensory arousal > 0.3 dampens starvation (-0.1)

`EmotionState.update()` receives `sensory_novelty`, `sensory_stability`, `speech_detected`:
- Sensory novelty > 0.8 amplifies surprise (+0.15 * novelty)
- Sensory stability > 3 ticks dampens arousal (*0.9)
- Speech detected nudges valence positive (+0.05)

### Layer 2: Sensory Consciousness

`GlobalWorkspace.update()` receives `sensory_novelty`, `binding_familiarity`:
- Sensory novelty > 0.7 boosts effective_r (+0.05 * novelty) for ignition threshold
- Binding familiarity > 0.7 strengthens broadcast intensity (*1.1)

`MeditationState.should_enter()` receives `sensory_stability`:
- Additional entry condition: both audio and vision stability > 2 (can't meditate in chaos)

### Layer 3: Sensory Narration

`organism._higher_order_reflect()` adds sensory stats to meta-reflection context.
`organism.self_reflect()` adds sensory summary to reflection prompt.
`organism.tick()` returns sensory_stats_line in psyche output for PFC access.

## Files Changed

| File | Change |
|------|--------|
| `halo3/psyche/drives.py` | Add sensory params to `update()` |
| `halo3/psyche/emotions.py` | Add sensory params to `update()` |
| `halo3/psyche/workspace.py` | Add sensory params to `update()` |
| `halo3/psyche/meditation.py` | Add sensory stability to entry check |
| `halo3/psyche/organism.py` | Pass sensory stats through; inject into reflection context |
| `halo3/main.py` | Parse sensory stats, pass to organism.tick() |
| `tests/senses/test_sensory_cross_integration.py` | New test file |

## VRAM Cost

Zero. All changes are Python-level arithmetic on existing float scalars.

## Constraints

- No new `@eqx.filter_jit` functions
- No new GPU allocations
- All coupling constants are hardcoded (not learned) — Option A from brainstorming
- Sensory stats are already computed; we only pass them to new consumers
