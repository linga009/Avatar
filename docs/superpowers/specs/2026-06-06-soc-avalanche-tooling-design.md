# SOC Avalanche Tooling: Persistence + Statistical Testing + Ablation

**Date:** 2026-06-06
**Status:** Design approved
**Context:** Supports the SOC avalanches paper. Independent of deferred memory pipeline.

## Problem

Avatar's avalanche data (sizes, durations, r history) is in-memory only — lost on every restart. The current MLE exponent estimates have no confidence intervals or goodness-of-fit testing. And there's no ablation control to prove power laws come from the SOC controller rather than the Kuramoto architecture itself.

## Design

Three independent pieces, all touching `cop.py` + `main.py` + `config.py`.

## Part 1: Persist Avalanche History

### Storage Format

JSON file at `data/checkpoints/avalanche_history.json`:

```json
{
  "sizes": [0.12, 0.45, 0.03, ...],
  "durations": [3, 8, 1, ...],
  "r_full_history": [0.45, 0.47, 0.42, ...],
  "r_median_ema": 0.48
}
```

`r_full_history` capped at last 5,000 entries to limit file size (~40KB at float precision).

### Methods on CriticalDynamics (cop.py)

```python
def save_avalanche_history(self, path: str) -> None:
    """Persist avalanche data to JSON."""
    import json
    data = {
        "sizes": self._avalanche_sizes,
        "durations": self._avalanche_durations,
        "r_full_history": self._r_full_history[-5000:],
        "r_median_ema": self._r_median_ema,
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_avalanche_history(self, path: str) -> None:
    """Load persisted avalanche data. Graceful no-op if file missing."""
    import json
    try:
        with open(path) as f:
            data = json.load(f)
        self._avalanche_sizes = data.get("sizes", [])
        self._avalanche_durations = data.get("durations", [])
        self._r_full_history = data.get("r_full_history", [])
        self._r_median_ema = data.get("r_median_ema", 0.5)
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # start fresh
```

### Integration in main.py

- **On startup** (after `organism = Organism(...)`): call `organism.cop.load_avalanche_history("data/checkpoints/avalanche_history.json")`
- **Every 100 ticks** (alongside knowledge graph save): call `organism.cop.save_avalanche_history("data/checkpoints/avalanche_history.json")`
- **Before dream** (alongside checkpoint save): call `organism.cop.save_avalanche_history("data/checkpoints/avalanche_history.json")`

## Part 2: Rigorous Statistical Testing

### Method on CriticalDynamics (cop.py)

```python
def avalanche_stats_rigorous(self) -> dict | None:
    """Clauset-Shalizi-Newman power-law testing. Returns None if n < 50."""
```

Runs when n >= 50, logged every 100 ticks alongside the quick stats.

### KS Goodness-of-Fit

Kolmogorov-Smirnov distance between empirical CDF and fitted power-law CDF:

```
D = max |F_empirical(x) - F_powerlaw(x)|
```

Where `F_powerlaw(x) = 1 - (x / x_min)^{-(tau-1)}` for continuous power law.

Small D = data is consistent with power law.

### Bootstrap p-value

1. Compute D_observed from real data
2. Generate 500 synthetic power-law samples (same n, same tau, same x_min)
3. Compute D for each synthetic sample
4. p = fraction of synthetic samples with D >= D_observed
5. p > 0.1 → cannot reject power-law hypothesis

500 samples (not 1000) to keep computation fast — runs inside the tick loop.

### Bootstrap Confidence Intervals

1. Resample `_avalanche_sizes` with replacement 500 times
2. Compute tau_MLE for each resample
3. 95% CI = [2.5th percentile, 97.5th percentile]
4. Same for alpha (durations) and sigma (branching ratio)

### Scaling Relation

For critical branching processes:

```
gamma = (tau - 1) / (alpha - 1)
```

Theoretical prediction depends on universality class. For mean-field branching: gamma = 0.5. Report the measured gamma and compare.

### Log Format

```
Avalanche (rigorous): n=50 τ=1.28 [1.15, 1.42] α=1.90 [1.72, 2.11] σ=1.08 [0.95, 1.22] | KS_size=0.12 p=0.34 | KS_dur=0.15 p=0.28 | γ=0.33
```

### Trigger

- Quick stats (`avalanche_stats`): n >= 20, every 100 ticks — unchanged
- Rigorous stats (`avalanche_stats_rigorous`): n >= 50, every 100 ticks — new

## Part 3: Ablation Control

### Config Addition

```python
# config.py
disable_soc_controller: bool = False
```

### Implementation in cop.py

In `__init__`:
```python
self._disable_soc = cfg.disable_soc_controller
```

In the SOC update path (inside `observe()`), before calling `_soc_update`:
```python
if self._tick <= self._warmup or self._disable_soc:
    K_aa_new, K_cc_new, K_cross_new = K_aa, K_cc, K_cross
else:
    K_aa_new, K_cc_new, K_cross_new = self._soc_update(...)
```

When disabled, K stays at whatever initial values are passed in. Avalanche detection still runs — we still measure avalanches, they just won't follow power laws (expected: exponential distribution).

### Running the Ablation

No new experiment runner code needed. Just run Avatar with the config override:

```python
# In a test or script:
cfg = Halo3Config(disable_soc_controller=True)
```

Or add to existing experiment configs in `experiments/configs.py`:
```python
SOC_OFF = {"disable_soc_controller": True}
```

## Files Changed

| File | Change |
|---|---|
| `halo3/psyche/cop.py` | `save_avalanche_history()`, `load_avalanche_history()`, `avalanche_stats_rigorous()`, `_disable_soc` flag |
| `halo3/main.py` | Call save/load on startup, every 100 ticks, before dream |
| `halo3/config.py` | `disable_soc_controller: bool = False` |

## What This Does NOT Change

- Kuramoto physics — untouched
- Emotion system, body-voice, chat — untouched
- Checkpoint format — untouched (avalanche history is a separate JSON file)
- Dream cycle — untouched (just adds a save call before dreaming)
- Existing `avalanche_stats` property — kept as-is (quick diagnostics at n≥20)
- Memory pipeline (deferred) — independent, no interaction

## Tests

- `test_avalanche_save_load_roundtrip`: save sizes/durations, load, verify match
- `test_avalanche_load_missing_file`: load nonexistent path, verify no crash, empty lists
- `test_avalanche_r_history_capped`: save with >5000 entries, load, verify capped at 5000
- `test_ks_power_law_accepted`: generate synthetic power-law data (n=100, tau=1.5), verify p > 0.1
- `test_ks_exponential_rejected`: generate synthetic exponential data (n=100), verify p < 0.1
- `test_bootstrap_ci_contains_true`: synthetic power-law (tau=1.5), verify 95% CI contains 1.5
- `test_scaling_relation`: synthetic data with known tau and alpha, verify gamma computed correctly
- `test_soc_disabled_k_unchanged`: set disable_soc_controller=True, call observe(), verify K_aa/K_cc/K_cross unchanged
- `test_soc_disabled_avalanches_still_detected`: verify avalanche detection runs even with SOC off
