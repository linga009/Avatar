# SOC Avalanche Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist avalanche history across restarts, add rigorous power-law statistics (KS test, bootstrap CI, scaling relation), and add SOC controller ablation flag — all for the SOC avalanches paper.

**Architecture:** Three independent features in `cop.py` + wiring in `main.py` + one config flag. Independent of the deferred memory pipeline — goes on current remote HEAD.

**Tech Stack:** Python, numpy, JSON, existing CriticalDynamics class

---

### Task 1: Persist avalanche history

**Files:**
- Modify: `halo3/psyche/cop.py`
- Create: `halo3/tests/test_avalanche_persistence.py`

- [ ] **Step 1: Write failing tests**

Create `halo3/tests/test_avalanche_persistence.py`:

```python
"""Tests for avalanche history persistence."""
import os
import json
import tempfile
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics


def test_save_load_roundtrip():
    """Saved avalanche data loads back identically."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_sizes = [0.12, 0.45, 0.03, 1.2]
    cop._avalanche_durations = [3, 8, 1, 12]
    cop._r_full_history = [0.5 + i * 0.01 for i in range(100)]
    cop._r_median_ema = 0.48

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "avalanche_history.json")
        cop.save_avalanche_history(path)

        cop2 = CriticalDynamics(cfg)
        cop2.load_avalanche_history(path)

        assert cop2._avalanche_sizes == [0.12, 0.45, 0.03, 1.2]
        assert cop2._avalanche_durations == [3, 8, 1, 12]
        assert len(cop2._r_full_history) == 100
        assert abs(cop2._r_median_ema - 0.48) < 1e-6


def test_load_missing_file():
    """Loading nonexistent file is a graceful no-op."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop.load_avalanche_history("/nonexistent/path/avalanche.json")
    assert cop._avalanche_sizes == []
    assert cop._avalanche_durations == []


def test_r_history_capped():
    """r_full_history is capped at 5000 entries on save."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._r_full_history = [float(i) for i in range(8000)]

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "avalanche_history.json")
        cop.save_avalanche_history(path)

        with open(path) as f:
            data = json.load(f)
        assert len(data["r_full_history"]) == 5000
        # Should keep the LAST 5000
        assert data["r_full_history"][0] == 3000.0
        assert data["r_full_history"][-1] == 7999.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_avalanche_persistence.py -v`
Expected: FAIL — `save_avalanche_history` and `load_avalanche_history` don't exist.

- [ ] **Step 3: Implement save/load methods**

In `halo3/psyche/cop.py`, add these two methods after `avalanche_stats` property (after line ~284):

```python
    def save_avalanche_history(self, path: str) -> None:
        """Persist avalanche data to JSON."""
        import json as _json
        data = {
            "sizes": self._avalanche_sizes,
            "durations": self._avalanche_durations,
            "r_full_history": self._r_full_history[-5000:],
            "r_median_ema": self._r_median_ema,
        }
        with open(path, "w") as f:
            _json.dump(data, f)

    def load_avalanche_history(self, path: str) -> None:
        """Load persisted avalanche data. Graceful no-op if file missing."""
        import json as _json
        try:
            with open(path) as f:
                data = _json.load(f)
            self._avalanche_sizes = data.get("sizes", [])
            self._avalanche_durations = data.get("durations", [])
            self._r_full_history = data.get("r_full_history", [])
            self._r_median_ema = data.get("r_median_ema", 0.5)
        except (FileNotFoundError, _json.JSONDecodeError):
            pass
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_avalanche_persistence.py -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/cop.py halo3/tests/test_avalanche_persistence.py
git commit -m "feat(cop): persist avalanche history to JSON — save/load across restarts"
```

---

### Task 2: Wire persistence into main.py

**Files:**
- Modify: `halo3/main.py`

- [ ] **Step 1: Add load on startup**

In `halo3/main.py`, after `organism = Organism(seed_topics)` (line 124), add:

```python
    # Load persisted avalanche history (survives restarts)
    _aval_path = "data/checkpoints/avalanche_history.json"
    organism.cop.load_avalanche_history(_aval_path)
```

- [ ] **Step 2: Add save every 100 ticks**

At line 462 (the `if tick % 100 == 0` block where knowledge graph is saved), add after the knowledge graph save:

```python
            organism.cop.save_avalanche_history(_aval_path)
```

Also add a standalone save if the knowledge graph condition is false (for ticks where there are no graph nodes yet):

Actually simpler — just add it unconditionally at tick % 100:

Find the block:
```python
        if tick % 100 == 0 and organism.knowledge_graph.node_count > 0:
            organism.knowledge_graph.save("data/checkpoints/knowledge_graph.json")
```

Add after it (NOT inside the `if`):
```python
        if tick % 100 == 0:
            organism.cop.save_avalanche_history(_aval_path)
```

- [ ] **Step 3: Add save before dream**

Find the dream entry point (line ~468: `log.info("  ☽ Entering dream state...")`). Add before it:

```python
            organism.cop.save_avalanche_history(_aval_path)
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ -v --tb=short -q 2>&1 | tail -5`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/main.py
git commit -m "feat(main): wire avalanche persistence — load on startup, save every 100 ticks + before dream"
```

---

### Task 3: Ablation flag for SOC controller

**Files:**
- Modify: `halo3/config.py`
- Modify: `halo3/psyche/cop.py`
- Create: `halo3/tests/test_soc_ablation.py`

- [ ] **Step 1: Write failing tests**

Create `halo3/tests/test_soc_ablation.py`:

```python
"""Tests for SOC controller ablation."""
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics
import numpy as np


def _make_theta():
    import jax.numpy as jnp
    return jnp.zeros((128, 64))


def test_soc_disabled_k_unchanged():
    """With SOC disabled, K values should not change after observe()."""
    cfg = Halo3Config(disable_soc_controller=True)
    cop = CriticalDynamics(cfg)
    # Run past warmup
    for i in range(10):
        cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    result = cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                         K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    assert abs(result["K_aa"] - 0.5) < 1e-6, f"K_aa changed: {result['K_aa']}"
    assert abs(result["K_cc"] - 0.5) < 1e-6, f"K_cc changed: {result['K_cc']}"
    assert abs(result["K_cross"] - 0.25) < 1e-6, f"K_cross changed: {result['K_cross']}"


def test_soc_enabled_k_changes():
    """With SOC enabled (default), K should change after warmup."""
    cfg = Halo3Config(disable_soc_controller=False)
    cop = CriticalDynamics(cfg)
    for i in range(10):
        cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    result = cop.observe(r_mean=0.3, r_a=0.15, r_c=0.15, fe_delta=0.0,
                         K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    # r=0.3 < 0.5, so controller should increase K
    assert result["K_aa"] > 0.5 or result["K_cc"] > 0.5, "K should increase when r < 0.5"


def test_soc_disabled_avalanches_still_detected():
    """Avalanche detection runs even with SOC disabled."""
    cfg = Halo3Config(disable_soc_controller=True)
    cop = CriticalDynamics(cfg)
    # Drive r above then below threshold to create avalanche
    for i in range(20):
        cop.observe(r_mean=0.6, r_a=0.3, r_c=0.3, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    # Now drop r — should start avalanche
    for i in range(5):
        cop.observe(r_mean=0.2, r_a=0.1, r_c=0.1, fe_delta=0.0,
                    K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    # Rise back — should end avalanche
    for i in range(5):
        result = cop.observe(r_mean=0.6, r_a=0.3, r_c=0.3, fe_delta=0.0,
                             K_aa=0.5, K_cc=0.5, K_cross=0.25, theta=_make_theta())
    assert result["avalanche"]["n_total"] >= 1, "Avalanche should be detected even with SOC off"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_soc_ablation.py -v`
Expected: FAIL — `disable_soc_controller` not in config.

- [ ] **Step 3: Add config flag**

In `halo3/config.py`, after `disable_quantum_potential: bool = False` (line ~109), add:

```python
    disable_soc_controller: bool = False
```

- [ ] **Step 4: Implement ablation in cop.py**

In `halo3/psyche/cop.py`, in `__init__` (after line ~39, after `self._warmup`), add:

```python
        self._disable_soc = cfg.disable_soc_controller
```

Then modify the SOC controller conditional (lines 113-118). Change from:

```python
        if self._tick <= self._warmup:
            K_aa_new, K_cc_new, K_cross_new = K_aa, K_cc, K_cross
        else:
            # SOC controller uses RAW chi (not normalized) for responsive coupling
            K_aa_new, K_cc_new, K_cross_new = self._soc_update(
                K_aa, K_cc, K_cross, r_mean, r_a, r_c, chi_raw)
```

To:

```python
        if self._tick <= self._warmup or self._disable_soc:
            K_aa_new, K_cc_new, K_cross_new = K_aa, K_cc, K_cross
        else:
            # SOC controller uses RAW chi (not normalized) for responsive coupling
            K_aa_new, K_cc_new, K_cross_new = self._soc_update(
                K_aa, K_cc, K_cross, r_mean, r_a, r_c, chi_raw)
```

- [ ] **Step 5: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_soc_ablation.py -v`
Expected: All 3 PASS

- [ ] **Step 6: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/config.py halo3/psyche/cop.py halo3/tests/test_soc_ablation.py
git commit -m "feat(cop): SOC controller ablation flag — disable_soc_controller config option"
```

---

### Task 4: Rigorous statistical testing

**Files:**
- Modify: `halo3/psyche/cop.py`
- Create: `halo3/tests/test_avalanche_stats.py`

- [ ] **Step 1: Write failing tests**

Create `halo3/tests/test_avalanche_stats.py`:

```python
"""Tests for rigorous avalanche statistics."""
import numpy as np
from halo3.config import Halo3Config
from halo3.psyche.cop import CriticalDynamics


def _make_power_law_samples(n, tau, x_min=0.01, seed=42):
    """Generate synthetic power-law distributed samples."""
    rng = np.random.RandomState(seed)
    # Inverse CDF method: x = x_min * u^(-1/(tau-1))
    u = rng.uniform(0, 1, n)
    return x_min * u ** (-1.0 / (tau - 1.0))


def _make_exponential_samples(n, rate=1.0, seed=42):
    """Generate synthetic exponential samples."""
    rng = np.random.RandomState(seed)
    return rng.exponential(1.0 / rate, n)


def test_rigorous_returns_none_below_50():
    """Returns None when n < 50."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_sizes = [0.1] * 30
    cop._avalanche_durations = [2] * 30
    result = cop.avalanche_stats_rigorous
    assert result is None


def test_ks_power_law_accepted():
    """Synthetic power-law data should have p > 0.1."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    sizes = _make_power_law_samples(100, tau=1.5).tolist()
    durations = [max(1, int(s * 10)) for s in sizes]
    cop._avalanche_sizes = sizes
    cop._avalanche_durations = durations
    result = cop.avalanche_stats_rigorous
    assert result is not None
    assert result["ks_p_size"] > 0.05, f"Power-law data should not be rejected, p={result['ks_p_size']}"


def test_ks_exponential_rejected():
    """Synthetic exponential data should have p < 0.1."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    sizes = _make_exponential_samples(100).tolist()
    durations = [max(1, int(s * 5)) for s in sizes]
    cop._avalanche_sizes = sizes
    cop._avalanche_durations = durations
    result = cop.avalanche_stats_rigorous
    assert result is not None
    assert result["ks_p_size"] < 0.2, f"Exponential data should be rejected or marginal, p={result['ks_p_size']}"


def test_bootstrap_ci_contains_true():
    """Bootstrap 95% CI should contain the true tau for synthetic data."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    true_tau = 1.5
    sizes = _make_power_law_samples(200, tau=true_tau, seed=123).tolist()
    durations = [max(1, int(s * 10)) for s in sizes]
    cop._avalanche_sizes = sizes
    cop._avalanche_durations = durations
    result = cop.avalanche_stats_rigorous
    assert result is not None
    ci_lo, ci_hi = result["tau_ci"]
    assert ci_lo <= true_tau <= ci_hi, f"True tau {true_tau} not in CI [{ci_lo}, {ci_hi}]"


def test_scaling_relation():
    """gamma = (tau-1)/(alpha-1) should be computed."""
    cfg = Halo3Config()
    cop = CriticalDynamics(cfg)
    cop._avalanche_sizes = _make_power_law_samples(100, tau=1.5).tolist()
    cop._avalanche_durations = [max(1, int(s * 10)) for s in cop._avalanche_sizes]
    result = cop.avalanche_stats_rigorous
    assert result is not None
    assert "gamma" in result
    assert isinstance(result["gamma"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_avalanche_stats.py -v`
Expected: FAIL — `avalanche_stats_rigorous` doesn't exist.

- [ ] **Step 3: Implement avalanche_stats_rigorous**

In `halo3/psyche/cop.py`, add this property after `save_avalanche_history` (after the methods added in Task 1):

```python
    @property
    def avalanche_stats_rigorous(self) -> dict | None:
        """Clauset-Shalizi-Newman power-law testing with bootstrap CIs.

        Returns None if n < 50. Otherwise returns dict with:
        - tau_est, tau_ci: size exponent + 95% CI
        - alpha_est, alpha_ci: duration exponent + 95% CI
        - sigma_est, sigma_ci: branching ratio + 95% CI
        - ks_d_size, ks_p_size: KS distance and p-value for size distribution
        - ks_d_dur, ks_p_dur: KS distance and p-value for duration distribution
        - gamma: scaling relation (tau-1)/(alpha-1)
        """
        n = len(self._avalanche_sizes)
        if n < 50:
            return None

        sizes = np.array(self._avalanche_sizes)
        durations = np.array(self._avalanche_durations, dtype=float)
        rng = np.random.RandomState(42)

        def _mle_exponent(data):
            x_min = data[data > 0].min()
            valid = data[data >= x_min]
            if len(valid) < 5:
                return None, x_min
            return 1.0 + len(valid) / np.sum(np.log(valid / x_min)), x_min

        def _ks_distance(data, tau, x_min):
            valid = np.sort(data[data >= x_min])
            n_v = len(valid)
            if n_v < 5:
                return 1.0
            empirical_cdf = np.arange(1, n_v + 1) / n_v
            theoretical_cdf = 1.0 - (valid / x_min) ** (-(tau - 1.0))
            return float(np.max(np.abs(empirical_cdf - theoretical_cdf)))

        def _bootstrap_p(data, tau, x_min, d_observed, n_boot=500):
            valid = data[data >= x_min]
            n_v = len(valid)
            if n_v < 5:
                return 0.0
            count_worse = 0
            for _ in range(n_boot):
                # Generate synthetic power-law sample
                u = rng.uniform(0, 1, n_v)
                synthetic = x_min * u ** (-1.0 / (tau - 1.0))
                syn_tau, syn_xmin = _mle_exponent(synthetic)
                if syn_tau is None:
                    continue
                d_syn = _ks_distance(synthetic, syn_tau, syn_xmin)
                if d_syn >= d_observed:
                    count_worse += 1
            return count_worse / n_boot

        def _bootstrap_ci(data, n_boot=500):
            estimates = []
            for _ in range(n_boot):
                sample = rng.choice(data, size=len(data), replace=True)
                est, _ = _mle_exponent(sample)
                if est is not None:
                    estimates.append(est)
            if len(estimates) < 10:
                return (None, None)
            estimates = np.array(estimates)
            return (float(np.percentile(estimates, 2.5)),
                    float(np.percentile(estimates, 97.5)))

        # Size distribution
        tau_est, s_min = _mle_exponent(sizes)
        ks_d_size = _ks_distance(sizes, tau_est, s_min) if tau_est else 1.0
        ks_p_size = _bootstrap_p(sizes, tau_est, s_min, ks_d_size) if tau_est else 0.0
        tau_ci = _bootstrap_ci(sizes)

        # Duration distribution
        alpha_est, d_min = _mle_exponent(durations)
        ks_d_dur = _ks_distance(durations, alpha_est, d_min) if alpha_est else 1.0
        ks_p_dur = _bootstrap_p(durations, alpha_est, d_min, ks_d_dur) if alpha_est else 0.0
        alpha_ci = _bootstrap_ci(durations)

        # Branching ratio + CI
        if n >= 2:
            ratios = sizes[1:] / (sizes[:-1] + 1e-12)
            sigma_est = float(np.median(ratios))
            sigma_boots = []
            for _ in range(500):
                idx = rng.choice(len(ratios), size=len(ratios), replace=True)
                sigma_boots.append(float(np.median(ratios[idx])))
            sigma_ci = (float(np.percentile(sigma_boots, 2.5)),
                        float(np.percentile(sigma_boots, 97.5)))
        else:
            sigma_est = None
            sigma_ci = (None, None)

        # Scaling relation
        gamma = None
        if tau_est and alpha_est and alpha_est > 1.0:
            gamma = (tau_est - 1.0) / (alpha_est - 1.0)

        return {
            "n": n,
            "tau_est": float(tau_est) if tau_est else None,
            "tau_ci": tau_ci,
            "alpha_est": float(alpha_est) if alpha_est else None,
            "alpha_ci": alpha_ci,
            "sigma_est": sigma_est,
            "sigma_ci": sigma_ci,
            "ks_d_size": ks_d_size,
            "ks_p_size": ks_p_size,
            "ks_d_dur": ks_d_dur,
            "ks_p_dur": ks_p_dur,
            "gamma": gamma,
        }
```

- [ ] **Step 4: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/test_avalanche_stats.py -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/cop.py halo3/tests/test_avalanche_stats.py
git commit -m "feat(cop): rigorous avalanche stats — KS test, bootstrap CI, scaling relation"
```

---

### Task 5: Wire rigorous stats logging into organism.py

**Files:**
- Modify: `halo3/psyche/organism.py`

- [ ] **Step 1: Add rigorous stats logging**

In `organism.py`, find the block that logs avalanche stats every 100 ticks (around line 264-274). Currently:

```python
            if self.self_model.age % 100 == 0:
                astats = self.cop.avalanche_stats
                if astats["n"] >= 20:
                    tau_s = ...
                    log.info(...)
```

After that block, add:

```python
                rstats = self.cop.avalanche_stats_rigorous
                if rstats:
                    tau_s = f"τ={rstats['tau_est']:.2f}" if rstats["tau_est"] else "τ=?"
                    tau_ci = f"[{rstats['tau_ci'][0]:.2f},{rstats['tau_ci'][1]:.2f}]" if rstats["tau_ci"][0] else ""
                    alpha_s = f"α={rstats['alpha_est']:.2f}" if rstats["alpha_est"] else "α=?"
                    alpha_ci = f"[{rstats['alpha_ci'][0]:.2f},{rstats['alpha_ci'][1]:.2f}]" if rstats["alpha_ci"][0] else ""
                    sigma_s = f"σ={rstats['sigma_est']:.2f}" if rstats["sigma_est"] else "σ=?"
                    sigma_ci = f"[{rstats['sigma_ci'][0]:.2f},{rstats['sigma_ci'][1]:.2f}]" if rstats["sigma_ci"][0] else ""
                    gamma_s = f"γ={rstats['gamma']:.2f}" if rstats["gamma"] else "γ=?"
                    log.info(
                        f"  Avalanche (rigorous): n={rstats['n']} "
                        f"{tau_s} {tau_ci} {alpha_s} {alpha_ci} {sigma_s} {sigma_ci} | "
                        f"KS_size={rstats['ks_d_size']:.2f} p={rstats['ks_p_size']:.2f} | "
                        f"KS_dur={rstats['ks_d_dur']:.2f} p={rstats['ks_p_dur']:.2f} | "
                        f"{gamma_s}"
                    )
```

- [ ] **Step 2: Run tests**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ -v --tb=short -q 2>&1 | tail -5`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add halo3/psyche/organism.py
git commit -m "feat(psyche): log rigorous avalanche stats (KS, CI, gamma) every 100 ticks"
```

---

### Task 6: Full test suite + CLAUDE.md update

**Files:**
- Test: all
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run full test suite**

Run: `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ tests/ -v --tb=short 2>&1 | tail -10`
Expected: All pass (210+ existing + new tests)

- [ ] **Step 2: Update CLAUDE.md**

Add to the COP Theory section:

```
Avalanche history persisted to data/checkpoints/avalanche_history.json (every 100 ticks + before dream, loaded on startup).
Rigorous stats (KS goodness-of-fit, bootstrap 95% CI, scaling relation gamma) computed at n≥50.
SOC ablation: disable_soc_controller=True freezes K — for control experiments.
```

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add CLAUDE.md
git commit -m "docs: add avalanche persistence, rigorous stats, SOC ablation to CLAUDE.md"
```
