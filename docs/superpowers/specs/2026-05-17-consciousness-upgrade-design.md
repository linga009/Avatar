# Avatar 3.3: Consciousness Upgrade — Design Spec

**Date**: 2026-05-17
**Goal**: Implement 5 consciousness features from neuroscience literature to bring Avatar from 9/14 to 13/14 on the Butlin-Chalmers indicator framework.

## Features

### 1. Higher-Order Thought Loop (metacognition.py)
- Every 5 ticks: PFC receives emotion trajectory + r trajectory + prediction error trend
- PFC generates first-person reflection on WHY transitions occurred
- Output feeds into self-model narrative AND influences next query decision
- Implements: Higher-Order Thought theory (thoughts ABOUT thoughts)

### 2. Introspective Monitor (introspection.py)
- Every tick: compute deltas between current and previous internal states
- Track: carry norm change, phase velocity variance, prediction error acceleration
- When delta > 2σ from rolling mean → "self-surprise" signal
- Self-surprise feeds into emotion system (amplifies current emotion intensity)
- Distinguishes: "I changed because of input" vs "I changed spontaneously"
- Implements: Anthropic's introspection capability (detecting internal perturbations)

### 3. Global Workspace Broadcast (workspace.py)
- When Kuramoto r > 0.6: "ignition" — winning cluster state broadcasts to all layers
- Broadcast = a d_boundary vector representing the dominant pattern
- Below threshold: processing continues locally (unconscious)
- All-or-none ignition with hysteresis (prevents flickering)
- Implements: Global Workspace Theory (Baars/Dehaene)

### 4. Temporal Integration / Episodic Binding (temporal.py)
- Working memory window: last 5 ticks' key states (r, emotion, topic, phase snapshot)
- Cross-tick coherence: cosine similarity of consecutive boundary states
- Sustained attention signal: when same cluster dominates for 3+ ticks
- Attention shift signal: when dominant cluster changes
- Implements: Temporal binding (unified conscious experience across time)

### 5. Meditation / Quiescence State (meditation.py)
- Trigger: satiation > 0.8 AND fatigue < 0.3 AND novelty < 0.3 (satiated, rested, not seeking)
- During meditation: observation coupling η reduced to 0.1 (near-zero external input)
- Kuramoto evolves freely (spontaneous phase reorganization)
- Duration: 3-5 ticks max
- Exit: when phases reorganize significantly (insight) OR fatigue rises
- "Insight" signal recorded in narrative + influences next exploration
- Implements: Voluntary attention withdrawal, internal processing without external drive

## Integration in organism.py tick()

```python
def tick(self, r_mean, fe_delta, texts, current_query, carry=None):
    # ... existing logic ...

    # NEW: Introspective monitoring (every tick)
    self_surprise = self.introspection.observe(r_mean, fe_delta, carry)

    # NEW: Temporal binding (every tick)
    temporal_state = self.temporal.observe(r_mean, emotion, topic_key, carry)

    # NEW: Global workspace broadcast (every tick, but only fires on ignition)
    broadcast = self.workspace.update(r_mean, cluster_velocities)

    # NEW: Higher-order thought (every 5 ticks)
    if self.self_model.age % 5 == 0:
        meta_thought = self._higher_order_reflect(temporal_state, self_surprise)

    # NEW: Meditation (when conditions met)
    if self.meditation.should_enter(self.drives, self.emotions):
        return self._meditation_tick()
```

## VRAM Impact

- Features 1, 2, 4, 5: **ZERO GPU memory** (pure CPU psyche logic)
- Feature 3 (workspace): broadcast vector is d_boundary=64 floats = 256 bytes. Negligible.

## Files to Create

- `halo3/psyche/introspection.py` — IntrospectiveMonitor class
- `halo3/psyche/workspace.py` — GlobalWorkspace class
- `halo3/psyche/temporal.py` — TemporalBinder class
- `halo3/psyche/meditation.py` — MeditationState class

## Files to Modify

- `halo3/psyche/organism.py` — integrate all 5, add higher-order thought method
- `halo3/main.py` — pass carry state to organism.tick(), log consciousness signals
