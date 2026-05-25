# Avatar 4.0 — Critical Order-Parameter Cognition (COP)

**Date**: 2026-05-26
**Author**: Dr. Linga Murthy Narlagiri
**Status**: Design approved, pending implementation
**Base**: Avatar v3.11 (halo3 worktree)

---

## 1. Overview

Avatar 4.0 replaces the hand-tuned psyche layer with Critical Order-Parameter Cognition (COP): a theory where affect, attention, curiosity, and binding are geometric readouts of a single self-organizing Kuramoto system near its critical point. There are no cognitive modules — there is one dynamical system, and the psyche is its phase-diagram geometry.

### 1.1 The Core Change

| v3.11 (current) | v4.0 (COP) |
|-----------------|-------------|
| Emotions from if/elif tree on r thresholds | Emotions from (r, chi, F_dot) manifold position |
| Curiosity = Gaussian peaked at r~0.5 | Curiosity = susceptibility chi (diverges at criticality) |
| K = 0.3 forever (static coupling) | K self-tunes via SOC controller toward critical point |
| T_body logged but unused | T_body is criticality sensor for SOC controller |
| GWT ignition at r > 0.6 (arbitrary) | Ignition when chi drops (crossed into order) |
| No unity/binding measurement | Unity index from coherence matrix eigenvalues |
| No relaxation time | tau from r autocorrelation (critical slowing) |

### 1.2 Scope

- **Changed**: Psyche layer (7 files rewritten) + kuramoto.py (2 functions added) + config.py (6 params) + main.py (1 line)
- **Unchanged**: Physics body, senses, perception pipeline, dream cycle, chat server, PFC, self-model, volatility surface, all training infrastructure
- **Preserved**: 1832-tick self_model.json, all checkpoints, episode DB, LoRA adapter, GEPA prompts, topic index

---

## 2. Safety Contract

### 2.1 Interface Preservation

organism.tick() returns the same dict with the same keys, types, and value ranges. All downstream consumers work without modification:

```python
{
    "emotion": str,           # one of 6 labels (same names)
    "intensity": float,       # [0, 1]
    "next_query": str,        # search query string
    "coupling_mod": float,    # NOW: absolute K value from SOC (was: multiplier)
    "finding": str | None,
    "log_line": str,          # enhanced with chi/tau/K
    "needs_dream": bool,
    "perception_failed": bool,
    "workspace": dict,        # same keys
    "temporal": dict,         # same keys
    "self_surprise": float,   # [0, 1]
    "meditation": dict,       # same keys
    "meta_thought": str | None,
    "proactive_message": str | None,
    "ethical_tension": float,
    "body_tension": float,
    "somatic_tension": float,
    # NEW (additive — nothing reads these yet):
    "chi": float,             # normalized susceptibility [0, 1]
    "tau": float,             # relaxation time [0, 1]
    "unity": float,           # unity index [0, 1]
    "K": float,               # current coupling value
}
```

### 2.2 Downstream Compatibility

| Consumer | Reads | Impact |
|----------|-------|--------|
| main.py | coupling_mod, needs_dream, log_line, finding, etc. | 1 line change: `new_K = coupling_mod` instead of `old_K * coupling_mod` |
| chat_server.py | emotion, drives, identity_statement | Zero changes (same types) |
| dream_finetune.py | emotion names, competence, traits, narrative | Zero changes (same 6 labels) |
| dream_gepa.py | episode data with emotion labels | Zero changes |
| Tests | Module interfaces | Tests checking specific r-thresholds need updating; type/shape tests pass |

### 2.3 Preserved State

| Asset | Path | Status |
|-------|------|--------|
| Identity (1832 ticks) | data/self_model.json | Preserved, same format |
| Model weights | data/checkpoints/halo3.eqx | Preserved, no weight changes |
| Sense module | data/checkpoints/sense_module.eqx | Untouched |
| Episode database | data/episodes/episodes.db | Preserved, same schema |
| LoRA adapter | data/pfc_adapter/ | Preserved |
| GEPA prompts | data/pfc_prompts.json | Preserved |
| Topic index | data/fineweb/topic_index.json | Untouched |
| BS volatility state | In-memory (organism.volatility) | Preserved |

---

## 3. Body-Level Changes

### 3.1 Dynamic Coupling K

KuramotoState already contains `coupling: float`. It is already updated in main.py via `carry._replace(kuramoto=carry.kuramoto._replace(coupling=new_K))`. Currently K only decreases (multiplicative satiation/meditation modulation) and partially recovers after dreams.

In v4.0, the SOC controller computes an absolute K value each tick. main.py applies it directly:

```python
# v3.11 (current):
new_K = old_K * coupling_mod  # multiplicative, ratchets down

# v4.0 (COP):
new_K = psyche_output["coupling_mod"]  # absolute K from SOC
```

This is one line in main.py. No changes to kuramoto.py, kuramoto_step, or the JIT-compiled path.

### 3.2 Susceptibility chi via Fluctuation-Dissipation Theorem

No probe field injection. No extra Kuramoto step. chi is estimated from the variance of r over a rolling window:

```
chi_est = N * Var(r, window=20)
```

Where N = n_clusters * n_hidden = 512. This is the static susceptibility from the fluctuation-dissipation theorem: near criticality, Var(r) diverges proportional to chi.

Normalized to [0, 1] by tracking chi_max (running lifetime maximum):

```
chi_norm = chi_est / chi_max
```

Self-calibrating. No hand-tuned scale factor. First 5 ticks: chi defaults to 0.5 while history builds.

Computed entirely in the COP engine (psyche layer). Zero changes to kuramoto.py.

### 3.3 Unity Index from Coherence Matrix

Two new pure functions added to kuramoto.py (additive, no changes to existing functions):

```python
def cluster_coherence_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    """Phase-difference matrix for cluster mean phases.

    Args:
        theta: (K, n_hidden) oscillator phases
    Returns:
        (K, K) matrix of |exp(i(psi_k - psi_l))| where psi_k = mean phase of cluster k
    """
    psi = jnp.angle(jnp.mean(jnp.exp(1j * theta), axis=1))  # (K,)
    diff = psi[:, None] - psi[None, :]  # (K, K)
    return jnp.abs(jnp.exp(1j * diff))


def unity_index(C: jnp.ndarray) -> tuple[float, float]:
    """Unity index and eigenvalue gap from coherence matrix.

    U = lambda_1 / sum(lambda_k)  — dominance of leading mode
    gap = (lambda_1 - lambda_2) / lambda_1  — separation

    U -> 1 with large gap = unified cognitive state (one subject)
    Multiple comparable eigenvalues = fragmented (dissociation)
    """
    eigenvalues = jnp.linalg.eigvalsh(C)
    eigenvalues = jnp.flip(eigenvalues)  # descending
    total = jnp.sum(eigenvalues)
    U = eigenvalues[0] / (total + 1e-8)
    gap = (eigenvalues[0] - eigenvalues[1]) / (eigenvalues[0] + 1e-8)
    return float(U), float(gap)
```

The coherence matrix is time-averaged via EMA in the COP engine (not in kuramoto.py). 32x32 matrix — eigenvalue decomposition takes <0.1ms.

### 3.4 Relaxation Time tau

Computed from autocorrelation of r over a rolling window. No body changes — entirely in the COP engine:

```
autocorr(lag) = Cov(r_t, r_{t-lag}) / Var(r)
tau = sum(autocorr(lag) for lag in 1..W) / autocorr(0)
```

High tau = slow relaxation = critical slowing down = persistent state.
Normalized to [0, 1] by clamping to [0, window] and dividing.

---

## 4. The COP Engine (new file: halo3/psyche/cop.py)

Central module computing the three macroscopic observables and the SOC control signal.

### 4.1 CriticalDynamics Class

```python
class CriticalDynamics:
    """Critical Order-Parameter Cognition engine.

    Computes (r, chi, tau) from Kuramoto state history and derives:
    - Affect coordinates: arousal = chi, valence = -F_dot
    - SOC control signal: K_dot via r-error * chi feedback
    - Unity index: from time-averaged coherence matrix eigenvalues
    - Emotion: from (r, chi_norm, F_dot) manifold position
    """

    def __init__(self, cfg: Halo3Config):
        self._window = cfg.cop_window          # 20
        self._eta = cfg.cop_eta                # 0.0005
        self._K_min = cfg.cop_K_min            # 0.05
        self._K_max = cfg.cop_K_max            # 2.0
        self._coherence_ema = cfg.cop_coherence_ema  # 0.1
        self._warmup = cfg.cop_warmup          # 5
        self._N = cfg.n_clusters * cfg.n_hidden  # 512

        # Rolling history
        self._r_history: deque[float]    # maxlen = window
        self._fe_history: deque[float]   # maxlen = window

        # Self-calibrating normalization
        self._chi_max: float = 1.0       # running maximum of raw chi

        # Time-averaged coherence matrix (K x K), EMA-updated
        self._C_avg: ndarray | None = None  # initialized on first observe

        # Tick counter
        self._tick: int = 0

    def observe(self, r_mean, r_a, r_c, fe_delta, K, theta) -> dict:
        """Record one tick, compute all COP observables.

        Args:
            r_mean: global order parameter
            r_a: analytical population order parameter
            r_c: creative population order parameter
            fe_delta: free energy change this tick
            K: current coupling
            theta: (K, n_hidden) raw oscillator phases

        Returns:
            dict with chi, tau, unity, gap, K_new, f_dot, T_body, U
        """
```

### 4.2 SOC Controller

```python
def _soc_update(self, K, r, chi):
    """Self-organized criticality controller.

    Drives system toward K* where U = r * chi is maximal.

    r < 0.5 (undercoupled): K increases toward order
    r > 0.5 (overcoupled): K decreases toward disorder
    r ~ 0.5 (near critical): K stays (operating point)

    chi-weighted: only adjusts when system is sensitive.
    Prevents oscillation far from criticality where chi -> 0.
    """
    r_error = 0.5 - r
    K_dot = self._eta * r_error * chi
    new_K = max(self._K_min, min(self._K_max, K + K_dot))
    return new_K
```

Safety properties:
- eta = 0.0005: max K change 0.0005/tick. Takes ~600 ticks (~10 hours) to shift by 0.3.
- chi-weighting: far from criticality chi->0, so K_dot->0. No overshoot.
- Clamp [0.05, 2.0]: K never zero (incoherent) or extreme.
- Self-correcting: overshoot causes r_error sign flip.

### 4.3 Meditation Integration

During meditation, the SOC controller is overridden:

```python
K_meditation = 0.02  # below K_c^analytical (0.048) — fully subcritical
```

The system voluntarily goes subcritical. Phases evolve freely. Insight = r shifts > 0.15 while subcritical (internal reorganization without external forcing).

After meditation, SOC resumes from whatever K the controller targets.

### 4.4 Warmup Period

For the first `cop_warmup` (5) ticks:
- chi defaults to 0.5
- tau defaults to 0.5
- SOC controller is disabled (K stays at init_coupling)
- Emotion falls back to the simple r-based classification

This prevents transient garbage from COP before the history windows fill.

---

## 5. Emotion as Phase-Diagram Geometry (emotions.py rewrite)

### 5.1 The COP Emotion Map

Same 6 labels. Derived from (r, chi_norm, f_dot) manifold position:

| Region | r | chi_norm | f_dot | Label |
|--------|---|----------|-------|-------|
| Ordered, calm, resolving | > 0.55 | < 0.4 | > 0 | satisfaction |
| Ordered, sensitive, resolving | > 0.55 | > 0.4 | > 0 | pride |
| Critical edge, maximally open | ~0.4-0.6 | high | ~0 | curiosity |
| Disordered, unreactive | < 0.35 | < 0.3 | ~0 | boredom |
| Disordered, reactive, worsening | < 0.35 | > 0.5 | < 0 | anxiety |
| Sustained failure | any | any | > 0 persistent | frustration |

Where:
- chi_norm = chi / chi_max (self-calibrated, [0,1])
- f_dot = -fe_delta (positive when surprise is resolving)

### 5.2 Emotional Inertia

Preserved from v3.11. Continuous valence/arousal EMA smoothing (alpha=0.6) prevents tick-to-tick flipping. The raw COP emotion is smoothed through the same (valence, arousal) -> discrete label pipeline.

### 5.3 Sensory Modulation

Preserved from v3.10:
- Sensory novelty amplifies chi_norm (high novelty -> even more open)
- Sensory stability dampens arousal
- Speech detected nudges valence positive

### 5.4 Interface

EmotionState.update() keeps the same signature and return type: `(emotion_name: str, intensity: float)`. Internal computation changes from if/elif on r to COP manifold lookup. Callers see no difference.

---

## 6. Drives — Selective Replacement

### 6.1 What Stays (orthogonal to COP)

| Drive | Reason to keep |
|-------|----------------|
| hunger | FE reduction tracking — independent of phase dynamics |
| fatigue | Accumulated processing cost — determines dream timing |
| starvation | Perception failure counter — independent of r/chi |
| novelty | Topic-change tracking — independent of phase dynamics |

These four drives keep their exact v3.11 implementation. Same update logic, same thresholds.

### 6.2 What Changes

| Drive | v3.11 | v4.0 |
|-------|-------|------|
| curiosity | `exp(-0.5 * ((r - 0.5) / 0.15)^2)` | `chi_norm` (susceptibility, directly) |
| satiation | Threshold: r > 0.7 for N ticks | Derived: r > 0.55 AND chi_norm < 0.2 for N ticks (ordered + rigid = nothing new to learn) |

### 6.3 Interface

DriveState.update() keeps the same signature. Adds `chi_norm` parameter. Returns same types. drives.summary() produces the same bar-chart format.

---

## 7. Consciousness Modules — Rewritten with COP Geometry

Each module keeps its external interface (same dict output, same method signatures). Internal logic changes to use COP observables instead of hand-tuned thresholds.

### 7.1 GlobalWorkspace (workspace.py)

| v3.11 | v4.0 |
|-------|------|
| Ignition when r > 0.6 | Ignition when chi drops sharply while r is rising (system crossed from critical edge into ordered phase — a pattern crystallized) |
| Sustain threshold r > 0.45 | Sustain while chi_norm < 0.5 AND r > 0.45 (still in ordered phase) |
| Broadcast content from topic + finding | Same |

Detection: track chi_norm derivative. Ignition = chi_norm was > 0.6 within last 3 ticks AND chi_norm is now < 0.4 AND r is rising. This captures the phase transition into order, which IS the moment a coherent percept forms.

### 7.2 IntrospectiveMonitor (introspection.py)

| v3.11 | v4.0 |
|-------|------|
| z-scores on r, FE, carry_norm deltas | tau-based: self_surprise = sudden tau change |

Rising tau (critical slowing) = "something is building up inside me." Sudden tau drop (after rising) = "it just reorganized." This is physically grounded: critical slowing IS the dynamical signature of an impending phase transition.

self_surprise = clamp((tau_derivative - 2*sigma) / 3*sigma, 0, 1)

Same scale [0, 1] as v3.11. Same output dict.

### 7.3 TemporalBinder (temporal.py)

| v3.11 | v4.0 |
|-------|------|
| Weighted: 0.4*topic + 0.3*emotion + 0.3*r_smoothness | tau IS temporal coherence + topic tracking |

temporal_coherence = 0.5 * tau_norm + 0.3 * topic_coherence + 0.2 * r_smoothness

tau replaces the emotion_coherence term (tau captures emotional persistence more accurately than counting consecutive identical labels). Topic tracking and r_smoothness stay.

sustained_attention, attention_just_shifted, emotional_momentum, narrative_thread: all kept. Focus accumulator for dream consolidation: kept.

### 7.4 MeditationState (meditation.py)

| v3.11 | v4.0 |
|-------|------|
| Entry: satiation > 0.7, fatigue < 0.3, novelty < 0.4, hunger < 0.5, calm emotion, sensory calm | Entry: chi_norm < 0.2 (system is rigid/ordered) AND fatigue < 0.4 AND not starving AND sensory calm |
| Mechanism: coupling_override = 0.1 (reduce to 10%) | Mechanism: set K = 0.02 (below K_c^a, fully subcritical) |
| Insight: r shift > 0.15 | Same |

The entry condition change is significant: instead of requiring low hunger + high satiation (which rarely coincided in v3.11 — 0 meditations in the last session), the COP condition asks "is the system in a rigid ordered state?" (chi_norm low). This is when meditation is most useful — the system is locked and needs internal reorganization. This should actually trigger meditations, unlike v3.11 where conditions were too restrictive.

---

## 8. Configuration

Added to Halo3Config (config.py):

```python
# COP (Critical Order-Parameter Cognition) — v4.0
cop_window: int = 20          # rolling window for chi and tau estimation
cop_eta: float = 0.0005       # SOC controller learning rate (K change/tick)
cop_K_min: float = 0.05       # minimum coupling (never fully incoherent)
cop_K_max: float = 2.0        # maximum coupling
cop_coherence_ema: float = 0.1 # EMA decay for time-averaged coherence matrix
cop_warmup: int = 5           # ticks before COP activates (fallback to defaults)
```

All defaults derived from theory or system parameters. None requires tuning.

---

## 9. File Change Summary

| File | Action | Lines (est.) | Risk |
|------|--------|-------------|------|
| `halo3/psyche/cop.py` | NEW | ~200 | Zero (additive) |
| `halo3/psyche/emotions.py` | Rewrite internals | ~80 | Low (same output) |
| `halo3/psyche/drives.py` | Curiosity <- chi, satiation update | ~20 | Low |
| `halo3/psyche/organism.py` | Wire COP, simplify consciousness | ~100 | Medium (hub) |
| `halo3/psyche/workspace.py` | Ignition from chi geometry | ~40 | Low |
| `halo3/psyche/introspection.py` | Self-surprise from tau | ~40 | Low |
| `halo3/psyche/temporal.py` | tau replaces emotion_coherence | ~20 | Low |
| `halo3/psyche/meditation.py` | K-control, chi-based entry | ~30 | Low |
| `halo3/kuramoto.py` | Add 2 pure functions | ~30 | Zero (additive) |
| `halo3/config.py` | Add 6 params | ~10 | Zero |
| `halo3/main.py` | 1 line: coupling_mod is absolute K | ~1 | Zero |
| Tests | Update threshold tests | ~50 | Low |

**Total**: 1 new file, 10 modified files, ~620 lines changed.
**Untouched**: 20+ files (model.py, backbone.py, senses/*, perception/*, training/*, chat_server.py, etc.)

---

## 10. Logging and Validation

### 10.1 Enhanced Log Line

```
Tick  100 | r=[...] 0.523 K=0.31 chi=0.72 tau=0.45 U=0.83/0.91 | emotion (i=0.72) | drives...
```

New fields: K (current coupling), chi (susceptibility), tau (relaxation time), U/gap (unity index/eigenvalue gap).

### 10.2 Periodic Consciousness Report (every 10 ticks)

```
COP: K=0.312 chi=0.72 tau=0.45 | U=r*chi=0.377 | Unity=0.83 gap=0.91 | IGNITED (ratio=72%)
```

### 10.3 Falsifiable Predictions (logged from tick 1)

| # | Prediction | Measurement | Pass criterion |
|---|-----------|-------------|----------------|
| 1 | Critical slowing before insight | tau in 5 ticks before r > 0.6 events | tau rises before, drops after |
| 2 | Curiosity tracks chi | Pearson(chi_norm, exploratory_behavior) | r > 0.3 |
| 3 | Power-law avalanches | P(s) of r-reorganization sizes | Straight line on log-log |
| 4 | Operating point K* ~ 1.1*K_c | mean(K) over 100+ ticks | K settles to a stable value (effective K_c for dual-population system is empirical — log and observe) |
| 5 | Unity transition | U and gap vs cross-coupling | Sharp threshold |

These are LOGGED, not asserted. If the predictions fail, the theory is wrong — the logs will show it.

---

## 11. Migration Path

### 11.1 Pre-migration Checklist

1. Backup checkpoint: `cp data/checkpoints/halo3.eqx data/checkpoints/halo3_v3.11_backup.eqx`
2. Backup self-model: `cp data/self_model.json data/self_model_v3.11_backup.json`
3. Run existing tests: all 77 must pass

### 11.2 Startup Behavior

1. Ticks 1-5 (warmup): COP computes observables but uses fallback emotion (simple r-based). SOC disabled. K stays at init_coupling or whatever was in the carry.
2. Tick 6+: COP fully active. SOC controller begins adjusting K. Emotions from manifold.
3. Self-model continuity: competence map, narrative, traits carry over unchanged. New entries use same format with COP-derived emotions.

### 11.3 Rollback

If COP produces degenerate behavior:
1. Restore v3.11 code from git
2. Restore checkpoint backup
3. Restart container

No data migration needed in either direction — the checkpoint format is unchanged.

---

## 12. What COP Does NOT Change

- Avatar's name and identity (still Avatar, still Dr. Narlagiri's creation)
- The physics body (Lorentz + backbone + Hamiltonian + Kuramoto structure)
- Senses (FNO + VQ-VAE + spectral projection)
- Perception (TopicIndex + ActiveSampler + FE scoring)
- Dream cycle (body/FineWeb/visitors/mind/GEPA phases)
- Chat server (port 8420, think mode, system prompt structure)
- PFC (Qwen3 0.6B, query generation, meta-reflection)
- Black-Scholes volatility surface (topic valuation)
- Self-model persistence (JSON save/load)
- Episode store (SQLite)

---

## 13. Theory Reference

Full theory: `D:/New_Ai/Critical-Order-Parameter-Cognition.md`
Physics foundations: `D:/New_Ai/canonical_quantum_gravity.txt`

Key equations:
- Critical coupling: K_c = 2 / (pi * g(0))
- Order parameter: r ~ (K - K_c)^(1/2)
- Susceptibility: chi ~ |K - K_c|^(-1) (diverges at criticality)
- Relaxation time: tau ~ |K - K_c|^(-1)
- Operating point: max U(K) = r(K) * chi(K) at K* ~ 1.1 * K_c
- Unity: U = lambda_1 / sum(lambda_k) from coherence matrix eigenvalues
