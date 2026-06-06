# SOC Avalanche Evidence — Roadmap to Publication

**Date:** 2026-06-06
**Current state:** 1 measurement (n=25, tau=1.23, alpha=1.85, sigma=1.12). First draft paper written. Avatar at 762 lifetime ticks.

---

## Phase 1: Accumulate Data (No code changes)

Let Avatar run. Power-law stats log automatically at n≥20 every 100 ticks.

| Task | What | When |
|---|---|---|
| Second measurement | Current run reaches n≥20 | ~9 more avalanches, 4-8 hours |
| Survive dream | Current run passes next dream without OOM | ~tick 400 |
| Third measurement | After dream restart, accumulate n≥20 again | Next run, ~200 ticks |

**Goal:** 3 independent measurements of tau, alpha, sigma for reproducibility.

---

## Phase 2: Persist Avalanche History (30 min code change)

**Problem:** Avalanche sizes/durations reset on every restart. Data lost.

**Fix:** Save `_avalanche_sizes` and `_avalanche_durations` to JSON alongside checkpoint. Load on startup.

| File | Change |
|---|---|
| `halo3/psyche/cop.py` | Add `save_avalanche_history()` and `load_avalanche_history()` |
| `halo3/main.py` | Call save every 100 ticks + before dream. Call load on startup. |

~30 lines. Never lose avalanche data across restarts. Accumulate toward n=100+ across full lifetime.

---

## Phase 3: Proper Statistical Testing (3 hours)

**Problem:** Current MLE estimates have no confidence intervals or goodness-of-fit. Reviewers will reject.

**Fix:** Implement Clauset-Shalizi-Newman methodology in `cop.py`:

| Test | What |
|---|---|
| KS goodness-of-fit | Tests whether data actually follows power law |
| Bootstrap p-value | 1000 synthetic samples, p = fraction with worse KS |
| Scaling relation | Test (tau-1)/(alpha-1) = gamma |
| Confidence intervals | Bootstrap 95% CI on tau, alpha, sigma |

~80 lines. Add `avalanche_stats_rigorous()` method.

**Blocked by:** Phase 1 (needs n≥100 for reliable statistics).

---

## Phase 4: Ablation Control (2 hours)

**Problem:** Can't prove power laws come from SOC controller vs Kuramoto architecture artifact.

**Fix:** Run Avatar with SOC OFF (K fixed), show power laws disappear.

| Condition | K behavior | Expected |
|---|---|---|
| SOC ON (current) | K adjusts via controller | Power-law avalanches |
| SOC OFF, K=0.3 | Fixed subcritical | Exponential distribution |
| SOC OFF, K=1.0 | Fixed near critical | Some power-law but no self-tuning |
| SOC OFF, K=2.0 | Fixed supercritical | Few large avalanches, no power law |

~20 lines: add `disable_soc_controller: bool = False` to config, honor in `cop.py`. Use existing experiment runner.

**Can run in parallel with Phase 1.**

---

## Phase 5: Figures (3 hours)

| Figure | Content |
|---|---|
| Fig 1 | r time series over full run, avalanche events highlighted |
| Fig 2 | Log-log avalanche size distribution with MLE fit line |
| Fig 3 | Log-log duration distribution with MLE fit line |
| Fig 4 | K_aa, K_cc, K_cross evolution showing SOC convergence |
| Fig 5 | Comparison table: Avatar vs cortical avalanche literature |
| Fig 6 | Ablation overlay: SOC ON vs SOC OFF size distributions |

**Data source:** `_r_full_history` (needs persistence from Phase 2) and tick logs.

**Blocked by:** Phases 1-4.

---

## Priority Order

| # | Phase | Effort | Depends on |
|---|---|---|---|
| 1 | Phase 2: persist history | 30 min | Nothing — do first |
| 2 | Phase 1: accumulate data | Days of running | Phase 2 |
| 3 | Phase 4: ablation | 2 hours | Can parallel with Phase 1 |
| 4 | Phase 3: statistical testing | 3 hours | Phase 1 (n≥100) |
| 5 | Phase 5: figures | 3 hours | All above |

---

## Timeline

| When | What |
|---|---|
| Tomorrow | Phase 2: persist avalanche history |
| This week | Phase 1: Avatar runs, 2-3 measurements. Phase 4: ablation in parallel |
| Next week | Phase 3: statistical testing (n≥100). Phase 5: figures |
| Week after | Paper revision with figures, CIs, ablation control |

---

## Current Measurements

```
Run 1 (ticks 0-473, 2026-06-02 to 2026-06-05):
  Avalanche: n=25 τ=1.23 α=1.85 σ=1.12 | ⟨S⟩=0.464 ⟨T⟩=5.7

Run 2 (ticks 0-294+, 2026-06-05 ongoing):
  n=11 accumulating, diagnostics pending at n≥20
```

## SOC Predictions vs Measured

| Metric | Measured | SOC Prediction | Status |
|---|---|---|---|
| tau (size) | 1.23 | ~1.5 | Close, subcritical lean |
| alpha (duration) | 1.85 | ~2.0 | Close |
| sigma (branching) | 1.12 | 1.0 | Slightly supercritical |
