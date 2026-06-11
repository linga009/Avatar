# Avatar Cerebellum: Forward Model for Action Consequence Prediction

**Date:** 2026-06-11
**Status:** Design — future implementation
**Context:** Avatar currently acts (selects topics, adjusts coupling) without previewing consequences. A cerebellum provides internal rehearsal — predicting where the body will go before committing.

## Problem

Avatar's organism selects actions (topic changes, exploration/exploitation, K modulation) purely from current state. There is no mechanism to ask "if I switch to topic X, what happens to r, chi, and my drives over the next N ticks?" This means:

1. **Topic selection is reactive** — BS valuation prices topics by past performance, but can't predict how the body will respond to a new topic's observation statistics.
2. **K trajectory is open-loop** — the SOC controller adjusts K proportionally to (0.5 - r), but doesn't anticipate whether a K change will produce a cascade or a gentle drift.
3. **Dream timing is heuristic** — fatigue threshold triggers dreams, but the system can't predict whether waiting 10 more ticks would complete a consolidation.

A cerebellum (forward model) would let Avatar internally simulate "what happens if..." before committing to an action, reducing costly mistakes in a system where each tick takes ~130s of gradient computation.

## Non-Goals

- **Motor control**: Avatar has no physical actuators. The cerebellum predicts cognitive/physics consequences, not motor trajectories.
- **Planning agent**: This is not a tree-search planner. It's a single-step or short-horizon (5-10 tick) predictor that informs action selection.
- **Replacing COP**: The cerebellum does not control coupling. It predicts what COP + Kuramoto will do, then the existing systems act.

## Design: Two-Layer Forward Model

The cerebellum has two complementary layers that operate at different timescales:

### Layer 1: Fast Physics Predictor (Analytical)

**What it does**: Linearizes the Kuramoto dynamics around the current state to predict r and chi over a short horizon (1-5 ticks) for a given K perturbation.

**How it works**: The Kuramoto order parameter dynamics near the current state can be approximated by the Jacobian of the mean-field equations. For the Kuramoto model with coupling K and frequency distribution g(ω):

```
dr/dt ≈ -r/τ_relax + K·r·(1 - r²)/2
```

where τ_relax depends on the frequency spread. This gives an analytical prediction without running the full 8192-oscillator simulation.

**Implementation**:
```python
class FastPredictor:
    """Linearized Kuramoto forward model — no learned parameters."""

    def predict_r(self, r_now, K_proposed, omega_std, n_ticks=5, dt=0.1):
        """Predict r trajectory for proposed K change.

        Uses mean-field ODE: dr/dt = -r/tau + K*r*(1-r²)/2
        where tau = 1/(pi*g(0)*K) near K_c.
        """
        r = r_now
        trajectory = [r]
        for _ in range(n_ticks):
            g0 = 1.0 / (math.pi * omega_std * math.sqrt(2 * math.pi))
            tau = 1.0 / (math.pi * g0 * max(K_proposed, 0.01) + 1e-8)
            drdt = -r / tau + K_proposed * r * (1 - r**2) / 2
            r = max(0.0, min(1.0, r + drdt * dt))
            trajectory.append(r)
        return trajectory

    def predict_chi(self, r_trajectory):
        """Estimate chi from predicted r variance."""
        if len(r_trajectory) < 3:
            return 0.0
        var_r = sum((x - sum(r_trajectory)/len(r_trajectory))**2
                    for x in r_trajectory) / len(r_trajectory)
        return min(1.0, 8192 * var_r)  # N * Var(r), clamped
```

**Strengths**: Zero-cost, no training, grounded in physics, available from tick 0.
**Weaknesses**: Mean-field approximation ignores quantum potential and block structure. Accuracy degrades far from current operating point.

### Layer 2: Slow Learned Predictor (MLP)

**What it does**: Learns the mapping from (state, action) → (Δr, Δchi, Δdrives) by training on Avatar's actual experience during dreams.

**Architecture**:
```
Input:  [r, r_a, r_c, chi, tau, K_aa, K_cc, K_cross,
         hunger, curiosity, satiation, action_embedding]  →  24 dims
Hidden: 64 → 64 (two layers, GELU activation)
Output: [Δr, Δchi, Δhunger, Δcuriosity, Δsatiation,
         avalanche_prob]  →  6 dims
```

~5K parameters. Trains during dream Phase 5 (GEPA) on experience buffer.

**Action encoding**: Actions are one of:
- `stay` (continue current topic): [1, 0, 0, 0]
- `explore` (switch to frontier topic): [0, 1, 0, 0]
- `exploit` (switch to high-value topic): [0, 0, 1, 0]
- `rest` (trigger dream): [0, 0, 0, 1]

Plus continuous features: proposed_K_delta (3 dims for aa/cc/cross changes).

**Training data**: Each tick, the organism records a transition tuple:
```python
(state_t, action_t, state_t+5)  # 5-tick lookahead
```

Stored in a ring buffer (capacity 2000). During dream GEPA phase, train the MLP for up to 50 steps with MSE loss + L2 regularization.

**Strengths**: Captures nonlinear interactions that the analytical model misses (quantum potential effects, block coupling dynamics, drive feedback loops).
**Weaknesses**: Needs ~200 ticks of experience before predictions are meaningful. Bootstrap problem during early life.

### Combining the Two Layers

The cerebellum blends both predictions with a learned gate:

```python
confidence = sigmoid(n_training_samples / 500 - 1)  # 0 early, ~1 after 1000 samples
prediction = (1 - confidence) * fast_prediction + confidence * learned_prediction
```

Early in life, the analytical predictor dominates. As the MLP accumulates training data, it gradually takes over. The gate is purely a function of sample count — no learned parameters.

## Integration Points

### 1. Topic Selection (organism.py)

Currently, topic selection in `_pick_topic()` uses BS valuation + drive state. The cerebellum adds an inner loop:

```python
def _pick_topic_with_preview(self, candidates):
    """Score candidates by predicted consequence, not just past value."""
    scored = []
    for topic in candidates:
        # Predict what happens if we switch to this topic
        action = self._encode_action(topic)
        pred = self.cerebellum.predict(self._current_state(), action)
        # Score: predicted chi gain + predicted hunger reduction
        score = pred.delta_chi * 0.6 + (-pred.delta_hunger) * 0.4
        scored.append((topic, score))
    return max(scored, key=lambda x: x[1])[0]
```

This does NOT replace BS valuation — it adds a forward-looking component. BS prices past experience; the cerebellum prices predicted future.

### 2. SOC Controller Preview (cop.py)

Before the SOC controller commits a K update, the cerebellum can preview the trajectory:

```python
# In CriticalDynamics._soc_update:
K_aa_proposed = K_aa + self._eta * (0.5 - r_a) * eff_chi + noise
r_predicted = self.cerebellum.fast.predict_r(r_mean, K_aa_proposed, omega_std=0.03)
# If predicted r overshoots badly, dampen the update
if r_predicted[-1] > 0.8 or r_predicted[-1] < 0.1:
    K_aa_proposed = K_aa + 0.3 * (K_aa_proposed - K_aa)  # 70% damping
```

This prevents the SOC controller from making aggressive K changes that would cause r to crash or saturate.

### 3. Dream Timing (organism.py)

Currently, dreams trigger on fatigue > 0.8. The cerebellum can predict whether delaying is beneficial:

```python
# Predict consequence of waiting vs dreaming now
wait_pred = self.cerebellum.predict(state, action="stay")
dream_pred = self.cerebellum.predict(state, action="rest")
# Delay dream if model predicts an avalanche completing
if wait_pred.avalanche_prob > 0.5 and self.drives.fatigue < 0.9:
    log.info("Cerebellum: delaying dream — avalanche in progress")
    return False  # don't dream yet
```

### 4. Experience Buffer (new: cerebellum.py)

```python
class ExperienceBuffer:
    """Ring buffer of (state, action, outcome) transitions for cerebellum training."""

    def __init__(self, capacity=2000):
        self._buffer = deque(maxlen=capacity)
        self._pending = None  # state recorded at tick t

    def record_state(self, state, action):
        """Called each tick — records current state + action taken."""
        if self._pending is not None:
            # Complete the previous transition with current state as outcome
            prev_state, prev_action, tick_recorded = self._pending
            if self._tick - tick_recorded >= 5:  # 5-tick lookahead
                self._buffer.append((prev_state, prev_action, state))
                self._pending = None
        self._pending = (state, action, self._tick)
        self._tick += 1

    def sample_batch(self, batch_size=32):
        """Random batch for MLP training during dreams."""
        ...
```

## File Layout

```
halo3/
  cerebellum.py          # FastPredictor + LearnedPredictor + ExperienceBuffer + Cerebellum
  psyche/organism.py     # Integration: _pick_topic_with_preview, dream timing
  psyche/cop.py          # Integration: SOC preview (optional, behind flag)
  config.py              # New fields: enable_cerebellum, cerebellum_horizon, cerebellum_lr
  tests/test_cerebellum.py
```

Single new file (`cerebellum.py`), plus minor integration hooks in organism.py and cop.py.

## Computational Budget

| Component | Cost per tick | Notes |
|-----------|--------------|-------|
| FastPredictor | <1ms | Pure arithmetic, no JAX |
| LearnedPredictor | ~5ms | 5K param MLP forward pass |
| Topic preview (4 candidates) | ~20ms | 4 × learned predict |
| SOC preview | ~1ms | FastPredictor only |
| **Total per-tick overhead** | **~25ms** | **< 0.02% of 130s tick** |
| Dream training (50 steps) | ~2s | One-time per dream |

Negligible overhead. The cerebellum is pure Python/NumPy — no JAX compilation needed for 5K params.

## Limitations and Risks

1. **Bootstrap problem**: The learned predictor needs ~200 ticks of experience before it's useful. During this period, only the analytical predictor is available. Mitigation: the confidence gate starts at 0.

2. **Distribution shift**: The MLP trains on past experience, but Avatar's dynamics change as K adapts and the knowledge graph grows. Mitigation: ring buffer with finite capacity ensures old transitions age out; dream retraining keeps the model current.

3. **Self-fulfilling prophecies**: If the cerebellum always predicts topic A is better, Avatar always picks A, and the MLP never learns about B. Mitigation: epsilon-greedy exploration (5% random topic selection) already exists in organism.py.

4. **Delayed consequences**: A topic switch might not show effects for 20+ ticks (slow FDT response). The 5-tick horizon captures immediate body response but misses long-term effects. Mitigation: could extend horizon to 10-20 ticks later, at the cost of noisier training signal.

5. **No motor system yet**: The action space is limited to {stay, explore, exploit, rest} + K adjustments. When Avatar gains motor capabilities, the action space and predictor architecture would need expansion.

## Implementation Roadmap

### Phase 1: FastPredictor only (2-3 hours)
- Implement `FastPredictor` in `cerebellum.py`
- Add SOC preview to `cop.py` (behind `enable_cerebellum` flag)
- Tests for analytical predictions vs known Kuramoto dynamics
- No learned component, no organism integration yet

### Phase 2: Experience buffer + learned predictor (3-4 hours)
- Implement `ExperienceBuffer` and `LearnedPredictor`
- Wire `record_state()` into organism tick
- Add MLP training to dream GEPA phase
- Checkpoint: `data/checkpoints/cerebellum_mlp.npz`

### Phase 3: Full integration (2-3 hours)
- `_pick_topic_with_preview()` in organism.py
- Dream timing preview
- Confidence gate blending
- Ablation config: `disable_cerebellum=True`

### Phase 4: Validation (ongoing)
- Compare topic selection quality with/without cerebellum
- Measure prediction accuracy (predicted Δr vs actual Δr)
- Tune horizon, confidence ramp, and action space

## Config Fields

```python
# Cerebellum (forward model) — v4.5
enable_cerebellum: bool = False        # master switch
cerebellum_horizon: int = 5            # prediction lookahead in ticks
cerebellum_lr: float = 0.001           # MLP learning rate
cerebellum_buffer_size: int = 2000     # experience ring buffer capacity
cerebellum_train_steps: int = 50       # MLP training steps per dream
cerebellum_confidence_scale: float = 500.0  # samples for 50% confidence
cerebellum_soc_damping: float = 0.7    # K update damping when r predicted extreme
```

All behind `enable_cerebellum=False` by default. Zero impact on existing behavior until explicitly enabled.

## Relationship to Existing Architecture

```
                    ┌──────────────┐
                    │  Cerebellum  │
                    │  (predict)   │
                    └──────┬───────┘
                           │ "what if?"
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐     ┌──────────────┐    ┌───────────┐
  │ Organism │     │ COP / SOC    │    │  Dreams   │
  │ (decide) │     │ (control K)  │    │ (timing)  │
  └────┬─────┘     └──────┬───────┘    └───────────┘
       │                  │
       ▼                  ▼
  ┌──────────────────────────────────┐
  │      Kuramoto + Hamiltonian      │
  │         (physics body)           │
  └──────────────────────────────────┘
```

The cerebellum sits ABOVE the physics, alongside the psyche. It reads state, predicts forward, and advises — but never directly modifies coupling, phases, or drives. All authority remains with the existing COP/SOC controller and organism decision logic.
