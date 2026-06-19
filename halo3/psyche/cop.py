"""Critical Order-Parameter Cognition (COP) engine.

The central thesis: affect, attention, curiosity, and binding are
geometric readouts of where a single self-organizing Kuramoto system
sits relative to its critical point. There are no cognitive modules.

Three macroscopic observables:
  r   — order parameter (integration/coherence)
  chi — susceptibility (openness/reactivity, IS curiosity)
  tau — relaxation time (persistence/critical slowing)

Proportional criticality controller: K_dot = eta * (0.5 - r) * chi
  Drives coupling K toward the critical point where U = r * chi is maximal.

Unity index: eigenvalue dominance of time-averaged coherence matrix.

Reference: Critical-Order-Parameter-Cognition.md
"""
from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

log = logging.getLogger(__name__)

from halo3.config import Halo3Config
from halo3.kuramoto import cluster_coherence_matrix, unity_index


class CriticalDynamics:
    """Computes (r, chi, tau) and criticality control signal each tick."""

    def __init__(self, cfg: Halo3Config) -> None:
        self._window = cfg.cop_window          # 50
        self._eta = cfg.cop_eta                # 0.05
        self._K_min = cfg.cop_K_min            # 0.05 (for K_cross)
        self._K_max = cfg.cop_K_max            # 2.0  (for K_cross)
        # Block-specific bounds bracket each population's critical coupling
        self._K_min_aa = cfg.cop_K_min_aa      # 0.02
        self._K_max_aa = cfg.cop_K_max_aa      # 0.40
        self._K_min_cc = cfg.cop_K_min_cc      # 0.20
        self._K_max_cc = cfg.cop_K_max_cc      # 2.00
        self._soc_noise = cfg.cop_soc_noise    # 0.5 (fraction of eta)
        self._boundary_repulsion = cfg.cop_boundary_repulsion  # 3.0
        self._coherence_ema = cfg.cop_coherence_ema  # 0.02
        self._warmup = cfg.cop_warmup          # 5
        self._N = cfg.n_clusters * cfg.n_hidden  # n_clusters * n_hidden
        self._enable_cerebellum = cfg.enable_cerebellum
        self._soc_damping = cfg.cerebellum_soc_damping

        self._r_history: deque[float] = deque(maxlen=self._window)
        self._fe_history: deque[float] = deque(maxlen=self._window)
        self._obs_norm_history: deque[float] = deque(maxlen=self._window)
        self._f_thermo_history: deque[float] = deque(maxlen=self._window)

        self._chi_max: float = 1.0
        self._C_avg: np.ndarray | None = None
        self._tick: int = 0
        self._dF_dt_ema: float = 0.0  # EMA of F_thermo rate of change

        # Avalanche detection state
        self._r_full_history: list[float] = []  # unbounded, for offline analysis
        self._avalanche_sizes: list[float] = []
        self._avalanche_durations: list[int] = []
        self._in_avalanche: bool = False
        self._aval_start: int = 0
        self._aval_accum: float = 0.0  # accumulated size of current avalanche
        self._r_median_ema: float = 0.5  # adaptive threshold (EMA of r)
        self._aval_shape: list[float] = []  # per-tick deficit of current avalanche
        self._avalanche_shapes: list[list[float]] = []  # completed avalanche shapes
        self._disable_soc: bool = cfg.disable_soc_controller

    def observe(
        self,
        r_mean: float,
        r_a: float,
        r_c: float,
        fe_delta: float,
        K_aa: float = 0.3,
        K_cc: float = 0.3,
        K_cross: float = 0.15,
        theta=None,
        obs_norm: float = 0.0,
        H_mean: float | None = None,
        # Legacy single-K parameter for backward compatibility
        K: float | None = None,
    ) -> dict:
        """Record one tick, compute all COP observables.

        Args:
            r_mean: global order parameter
            r_a: analytical population order parameter
            r_c: creative population order parameter
            fe_delta: free energy change this tick
            K_aa: analytical self-coupling
            K_cc: creative self-coupling
            K_cross: cross-population coupling
            theta: (K_clusters, n_hidden) raw oscillator phases (JAX array)
            obs_norm: norm of observation vector (for drive-variance correction)
            K: legacy single coupling (ignored if K_aa/K_cc/K_cross provided)

        Returns:
            dict with chi, tau, unity, gap, K_aa, K_cc, K_cross, K_new,
            f_dot, T_body, U_product
        """
        # Legacy support: if only K is provided, use it for all three
        if K is not None and K_aa == 0.3 and K_cc == 0.3 and K_cross == 0.15:
            K_aa = K
            K_cc = K
            K_cross = K * 0.5

        self._tick += 1
        self._r_history.append(r_mean)
        self._fe_history.append(fe_delta)
        self._obs_norm_history.append(obs_norm)

        chi_raw, chi_norm = self._compute_chi()
        tau = self._compute_tau()
        T_body = abs(r_a - r_c)
        f_dot = -fe_delta
        import jax.numpy as jnp
        _theta = theta if theta is not None else jnp.zeros((1, 1))
        unity_val, gap = self._update_unity(_theta)

        if self._tick <= self._warmup or self._disable_soc:
            K_aa_new, K_cc_new, K_cross_new = K_aa, K_cc, K_cross
        else:
            # SOC controller uses RAW chi (not normalized) for responsive coupling
            K_aa_new, K_cc_new, K_cross_new = self._soc_update(
                K_aa, K_cc, K_cross, r_mean, r_a, r_c, chi_raw)

        U_product = r_mean * chi_norm

        # Thermodynamic diagnostic: Helmholtz free energy
        F_thermo = None
        if H_mean is not None and theta is not None:
            # Phase entropy from cluster mean phases (histogram-based)
            psi = jnp.angle(jnp.mean(jnp.exp(1j * _theta), axis=1))  # (K,)
            n_bins = 16
            # Manual histogram for JAX compatibility
            bin_edges = jnp.linspace(-jnp.pi, jnp.pi, n_bins + 1)
            counts = jnp.zeros(n_bins)
            for b in range(n_bins):
                counts = counts.at[b].set(
                    jnp.sum((psi >= bin_edges[b]) & (psi < bin_edges[b + 1]))
                )
            probs = counts / (jnp.sum(counts) + 1e-12)
            S_phase = float(-jnp.sum(probs * jnp.log(probs + 1e-12)))

            # Effective temperature from COP observables (uses raw chi)
            T_eff = chi_raw * (1.0 + tau)

            F_thermo = float(H_mean) - T_eff * S_phase

        # Track F_thermo rate of change (EMA)
        dF_dt = 0.0
        if F_thermo is not None:
            if len(self._f_thermo_history) > 0:
                dF_dt = F_thermo - self._f_thermo_history[-1]
                alpha_f = 0.3  # fast EMA for responsiveness
                self._dF_dt_ema = alpha_f * dF_dt + (1.0 - alpha_f) * self._dF_dt_ema
            self._f_thermo_history.append(F_thermo)
            dF_dt = self._dF_dt_ema

        # Avalanche detection on r
        aval_info = self._detect_avalanche(r_mean)

        return {
            "chi": chi_norm,       # normalized [0,1] — for emotions, display
            "chi_raw": chi_raw,    # unnormalized N*Var(r) — for SOC controller
            "tau": tau,
            "unity": unity_val,
            "gap": gap,
            "K_aa": K_aa_new,
            "K_cc": K_cc_new,
            "K_cross": K_cross_new,
            "K_new": (K_aa_new + K_cc_new + K_cross_new) / 3,  # backward compat
            "f_dot": f_dot,
            "T_body": T_body,
            "U_product": U_product,
            "F_thermo": F_thermo,
            "dF_dt": dF_dt,
            "avalanche": aval_info,
        }

    def post_dream_reset(self, terminal_r: float) -> None:
        """Reset transient COP state after a dream cycle.

        - Pre-fills r_history with noisy terminal_r to prevent variance collapse
        - Resets C_avg coherence matrix (stale after phase changes)
        - Does NOT reset chi_max (it decays naturally via 0.995 factor)

        The noise (±0.02) gives Var(r) ≈ 1.3e-4, so chi_raw ≈ N*Var ≈ 1.1
        instead of zero. Without this, chi stays at 0 for ~40 ticks while
        the 50-tick window fills with naturally varied r values — killing
        the SOC controller and preventing consciousness ignition.
        """
        import random as _rng
        self._r_history.clear()
        for _ in range(min(10, self._window)):
            self._r_history.append(terminal_r + _rng.uniform(-0.02, 0.02))
        # Reset coherence matrix — it will re-accumulate from fresh phases
        self._C_avg = None

    # ------------------------------------------------------------------
    # Avalanche detection — r excursions below adaptive threshold
    # ------------------------------------------------------------------

    def _detect_avalanche(self, r: float) -> dict:
        """Track avalanches as contiguous excursions of r below threshold.

        Uses an EMA of r as an adaptive threshold. An avalanche starts when
        r drops below the threshold and ends when it rises back above.
        Size = cumulative deficit (threshold - r) over the avalanche.
        Duration = number of ticks below threshold.

        Returns dict with current avalanche state and running totals.
        """
        self._r_full_history.append(r)

        # Adaptive threshold: slow EMA of r (tau ~ 100 ticks)
        alpha_thresh = 0.01
        self._r_median_ema = alpha_thresh * r + (1.0 - alpha_thresh) * self._r_median_ema

        thresh = self._r_median_ema
        below = r < thresh

        # Use r_full_history length as internal tick counter so _detect_avalanche
        # works correctly whether called via observe() or directly (e.g. in tests).
        internal_tick = len(self._r_full_history)

        just_ended = False
        if below and not self._in_avalanche:
            # Avalanche starts
            self._in_avalanche = True
            self._aval_start = internal_tick
            deficit = thresh - r
            self._aval_accum = deficit
            self._aval_shape = [deficit]
        elif below and self._in_avalanche:
            # Avalanche continues
            deficit = thresh - r
            self._aval_accum += deficit
            self._aval_shape.append(deficit)
        elif not below and self._in_avalanche:
            # Avalanche ends — record it
            self._in_avalanche = False
            duration = internal_tick - self._aval_start
            if duration >= 1 and self._aval_accum > 0:
                self._avalanche_sizes.append(self._aval_accum)
                self._avalanche_durations.append(duration)
                self._avalanche_shapes.append(self._aval_shape[:])
                just_ended = True
            self._aval_shape = []

        return {
            "in_avalanche": self._in_avalanche,
            "just_ended": just_ended,
            "n_total": len(self._avalanche_sizes),
            "threshold": thresh,
        }

    @property
    def avalanche_stats(self) -> dict:
        """Compute power-law exponent estimates for accumulated avalanches.

        Uses simple log-log regression as a quick diagnostic.
        For rigorous analysis, use the powerlaw package offline.
        """
        n = len(self._avalanche_sizes)
        if n < 20:
            return {"n": n, "tau_est": None, "alpha_est": None, "branching_est": None}

        sizes = np.array(self._avalanche_sizes)
        durations = np.array(self._avalanche_durations)

        # Quick tau estimate: MLE for power law on sizes
        # For P(x) ~ x^{-tau}, MLE gives tau = 1 + n / sum(ln(x/xmin))
        s_min = sizes[sizes > 0].min()
        valid = sizes >= s_min
        if valid.sum() > 5:
            tau_est = 1.0 + valid.sum() / np.sum(np.log(sizes[valid] / s_min))
        else:
            tau_est = None

        # Quick alpha estimate: MLE on durations
        d_min = max(1, durations[durations > 0].min())
        valid_d = durations >= d_min
        if valid_d.sum() > 5:
            alpha_est = 1.0 + valid_d.sum() / np.sum(np.log(durations[valid_d] / d_min))
        else:
            alpha_est = None

        # Branching ratio: ratio of consecutive avalanche sizes
        # (simplified — proper σ needs per-tick event counts)
        if n >= 2:
            ratios = sizes[1:] / (sizes[:-1] + 1e-12)
            branching_est = float(np.median(ratios))
        else:
            branching_est = None

        return {
            "n": n,
            "tau_est": float(tau_est) if tau_est is not None else None,
            "alpha_est": float(alpha_est) if alpha_est is not None else None,
            "branching_est": branching_est,
            "mean_size": float(sizes.mean()),
            "mean_duration": float(durations.mean()),
            "sizes": self._avalanche_sizes,
            "durations": self._avalanche_durations,
        }

    def save_avalanche_history(self, path: str) -> None:
        """Persist avalanche data to JSON."""
        import json as _json
        data = {
            "sizes": self._avalanche_sizes,
            "durations": self._avalanche_durations,
            "r_full_history": self._r_full_history[-5000:],
            "r_median_ema": self._r_median_ema,
            "shapes": self._avalanche_shapes[-200:],  # cap at 200 most recent
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
            self._avalanche_shapes = data.get("shapes", [])
        except (FileNotFoundError, _json.JSONDecodeError):
            pass

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

        def _log_likelihood_power_law(data, tau, x_min):
            """Log-likelihood of data under power-law with exponent tau."""
            valid = data[data >= x_min]
            n = len(valid)
            if n < 5 or tau <= 1.0:
                return -np.inf
            return n * np.log(tau - 1) - n * np.log(x_min) - tau * np.sum(np.log(valid / x_min))

        def _log_likelihood_lognormal(data, x_min):
            """Log-likelihood of data under log-normal (MLE fit)."""
            valid = data[data >= x_min]
            if len(valid) < 5:
                return -np.inf
            log_valid = np.log(valid)
            mu = np.mean(log_valid)
            sigma = np.std(log_valid)
            if sigma < 1e-12:
                return -np.inf
            return np.sum(-0.5 * ((log_valid - mu) / sigma) ** 2
                          - np.log(sigma) - 0.5 * np.log(2 * np.pi) - log_valid)

        def _log_likelihood_exponential(data, x_min):
            """Log-likelihood of data under exponential (MLE fit)."""
            valid = data[data >= x_min]
            if len(valid) < 5:
                return -np.inf
            lam = 1.0 / (np.mean(valid) - x_min + 1e-12)
            return len(valid) * np.log(lam) - lam * np.sum(valid - x_min)

        tau_est, s_min = _mle_exponent(sizes)
        ks_d_size = _ks_distance(sizes, tau_est, s_min) if tau_est else 1.0
        ks_p_size = _bootstrap_p(sizes, tau_est, s_min, ks_d_size) if tau_est else 0.0
        tau_ci = _bootstrap_ci(sizes)

        alpha_est, d_min = _mle_exponent(durations)
        ks_d_dur = _ks_distance(durations, alpha_est, d_min) if alpha_est else 1.0
        ks_p_dur = _bootstrap_p(durations, alpha_est, d_min, ks_d_dur) if alpha_est else 0.0
        alpha_ci = _bootstrap_ci(durations)

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

        gamma = None
        if tau_est and alpha_est and alpha_est > 1.0:
            gamma = (tau_est - 1.0) / (alpha_est - 1.0)

        # Alternative distribution comparison (Clauset-Shalizi-Newman 2009)
        ll_pl = _log_likelihood_power_law(sizes, tau_est, s_min) if tau_est else -np.inf
        ll_ln = _log_likelihood_lognormal(sizes, s_min)
        ll_exp = _log_likelihood_exponential(sizes, s_min)

        # Likelihood ratios: positive = power-law preferred
        lognormal_lr = ll_pl - ll_ln
        exponential_lr = ll_pl - ll_exp

        # Determine preferred model
        if lognormal_lr > 0 and exponential_lr > 0:
            preferred = "power_law"
        elif ll_ln > ll_pl and ll_ln > ll_exp:
            preferred = "lognormal"
        else:
            preferred = "exponential"

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
            "lognormal_lr": float(lognormal_lr) if np.isfinite(lognormal_lr) else 0.0,
            "exponential_lr": float(exponential_lr) if np.isfinite(exponential_lr) else 0.0,
            "preferred_model": preferred,
        }

    @property
    def shape_collapse_quality(self) -> dict | None:
        """Avalanche shape collapse analysis.

        Rescales each avalanche profile to normalized time [0,1] and
        amplitude, then measures how well they collapse onto a single
        universal curve. Good collapse (low error) supports criticality.

        Returns None if fewer than 10 shapes with duration >= 3.
        """
        # Filter shapes with duration >= 3 (too short = noisy)
        valid = [(s, d) for s, d in zip(self._avalanche_shapes, self._avalanche_durations)
                 if d >= 3 and len(s) >= 3]

        if len(valid) < 10:
            return None

        n_bins = 20  # normalized time bins

        # Rescale each shape to [0,1] time and unit area
        rescaled = []
        for shape, dur in valid:
            arr = np.array(shape, dtype=float)
            area = np.sum(arr) + 1e-12
            # Normalize: time to [0,1], amplitude so area = 1
            t_orig = np.linspace(0, 1, len(arr))
            t_bins = np.linspace(0, 1, n_bins)
            # Interpolate to common grid
            interp = np.interp(t_bins, t_orig, arr / area)
            rescaled.append(interp)

        rescaled = np.array(rescaled)  # (n_shapes, n_bins)

        # Universal curve = mean of all rescaled shapes
        mean_curve = np.mean(rescaled, axis=0)

        # Collapse error: mean squared deviation from universal curve
        deviations = rescaled - mean_curve[None, :]
        collapse_error = float(np.mean(deviations ** 2))

        # Normalized collapse error (relative to variance of mean curve)
        mean_var = float(np.var(mean_curve))
        normalized_error = collapse_error / (mean_var + 1e-12)

        return {
            "collapse_error": collapse_error,
            "normalized_error": normalized_error,
            "n_shapes_used": len(valid),
            "mean_shape": mean_curve.tolist(),
        }

    def _compute_chi(self) -> tuple[float, float]:
        """Susceptibility with Harada-Sasa FDT-violation correction.

        Returns (chi_raw, chi_norm):
          chi_raw = N * Var(r) after Harada-Sasa correction (for SOC controller)
          chi_norm = chi_raw / chi_max, in [0, 1] (for emotions/display)

        chi_max decays at 0.995/tick (~140 tick half-life) to prevent
        permanent suppression from transient spikes.
        """
        if len(self._r_history) < 5:
            return 1.0, 0.5

        r_arr = list(self._r_history)
        n = len(r_arr)
        mean_r = sum(r_arr) / n
        var_r = sum((x - mean_r) ** 2 for x in r_arr) / n

        chi_raw = self._N * var_r

        # Harada-Sasa correction
        centered_r = [x - mean_r for x in r_arr]
        if var_r > 1e-12 and len(self._obs_norm_history) >= n:
            obs_arr = list(self._obs_norm_history)[-n:]
            mean_obs = sum(obs_arr) / n
            centered_obs = [x - mean_obs for x in obs_arr]
            var_obs = sum(x * x for x in centered_obs) / n

            # C(1): autocorrelation of r at lag 1
            c1 = sum(centered_r[i] * centered_r[i - 1] for i in range(1, n)) / (n - 1)
            c1 /= (var_r + 1e-12)

            # R(1): cross-correlation of r with obs at lag 1
            if var_obs > 1e-12:
                r1 = sum(centered_r[i] * centered_obs[i - 1]
                         for i in range(1, n)) / (n - 1)
                r1 /= ((var_r * var_obs) ** 0.5 + 1e-12)
            else:
                r1 = 0.0

            # FDT violation: sigma >= 0
            sigma = max(0.0, c1 - r1)
            chi_raw = chi_raw / (1.0 + sigma * 5.0)

        # Decaying max — forgets old spikes over ~140 ticks
        self._chi_max = max(chi_raw, self._chi_max * 0.995)
        chi_norm = min(1.0, chi_raw / (self._chi_max + 1e-12))

        return chi_raw, chi_norm

    def _compute_tau(self) -> float:
        """Relaxation time from autocorrelation of r.

        High tau = critical slowing = persistent state.
        Normalized to [0, 1].
        """
        if len(self._r_history) < 5:
            return 0.5

        r_arr = list(self._r_history)
        n = len(r_arr)
        mean_r = sum(r_arr) / n
        centered = [x - mean_r for x in r_arr]

        var = sum(x * x for x in centered) / n
        if var < 1e-12:
            return 1.0

        max_lag = min(n // 2, 10)
        tau_sum = 0.0
        for lag in range(1, max_lag + 1):
            cov = sum(centered[i] * centered[i - lag] for i in range(lag, n)) / (n - lag)
            autocorr = cov / var
            if autocorr < 0:
                break
            tau_sum += autocorr

        return min(1.0, tau_sum / max_lag)

    def _soc_update(self, K_aa: float, K_cc: float, K_cross: float,
                    r: float, r_a: float, r_c: float, chi: float) -> tuple[float, float, float]:
        """Proportional criticality controller — block coupling.

        Three independent controllers with block-specific K bounds:
          K_aa: analytical self-coupling, targets r_a ~ 0.5, bounds [0.02, 0.40]
          K_cc: creative self-coupling, targets r_c ~ 0.5, bounds [0.20, 2.00]
          K_cross: cross-coupling, targets global r ~ 0.5, bounds [0.05, 2.0]

        Anti-clamp-lock mechanisms:
          1. Strong stochastic noise (0.5 * eta) — real SOC needs continuous drive
          2. Boundary repulsion — 3x extra noise when K within 10% of bound
          3. Eta attenuation at bounds — proportional term fades near clamps
          4. Floor-lock protection — zero proportional drive when pushing into floor
          5. Block imbalance correction — bias toward midpoint when blocks diverge
        """
        import random
        # chi is raw (N*Var(r)), typically 1-50. Floor of 1.0 lets
        # the controller bootstrap when fluctuations are very small.
        eff_chi = max(chi, 1.0)

        def _update_one(K: float, r_block: float, K_min: float, K_max: float) -> float:
            K_range = K_max - K_min
            proportional = 0.5 - r_block  # positive = increase K, negative = decrease K

            # Boundary repulsion — extra noise near bounds
            dist_to_min = (K - K_min) / K_range  # 0 at min, 1 at max
            dist_to_max = (K_max - K) / K_range  # 1 at min, 0 at max
            near_bound = min(dist_to_min, dist_to_max)  # 0 at either bound
            repulsion = 1.0 + self._boundary_repulsion * max(0, 1.0 - near_bound / 0.1)

            # Stronger base noise * repulsion at boundaries
            noise = self._eta * self._soc_noise * repulsion * (2.0 * random.random() - 1.0)

            # Attenuate eta when near clamp — let noise dominate at boundaries
            eta_scale = max(0.2, min(1.0, near_bound / 0.1))
            eff_eta = self._eta * eta_scale

            # Floor/ceiling-lock recovery (June 14-17 2026 fix):
            # When K is in the bottom 30% AND proportional pushes it lower,
            # kill the proportional and add a chi-scaled recovery that pushes
            # K back toward the 30% mark. Uses eff_chi so recovery strength
            # matches the proportional drive it replaces.
            recovery = 0.0
            if dist_to_min < 0.3 and proportional < 0:
                proportional = 0.0
                recovery = self._eta * (0.3 - dist_to_min) * eff_chi * 0.5
            elif dist_to_max < 0.3 and proportional > 0:
                proportional = 0.0
                recovery = -self._eta * (0.3 - dist_to_max) * eff_chi * 0.5

            K_new = K + eff_eta * proportional * eff_chi + noise + recovery
            return max(K_min, min(K_max, K_new))

        K_aa_new = _update_one(K_aa, r_a, self._K_min_aa, self._K_max_aa)
        K_cc_new = _update_one(K_cc, r_c, self._K_min_cc, self._K_max_cc)
        K_cross_new = _update_one(K_cross, r, self._K_min, self._K_max)

        # Block imbalance correction: when one block is at floor and the
        # other at ceiling, push both toward their midpoints. This breaks
        # the K_aa~0.02 / K_cc~2.0 asymmetric trap.
        aa_frac = (K_aa_new - self._K_min_aa) / (self._K_max_aa - self._K_min_aa)
        cc_frac = (K_cc_new - self._K_min_cc) / (self._K_max_cc - self._K_min_cc)
        imbalance = abs(aa_frac - cc_frac)
        if imbalance > 0.6:
            # Nudge each block toward its midpoint by 10% of the imbalance
            aa_mid = 0.5 * (self._K_min_aa + self._K_max_aa)
            cc_mid = 0.5 * (self._K_min_cc + self._K_max_cc)
            correction = 0.1 * imbalance * self._eta
            K_aa_new += correction * (aa_mid - K_aa_new) / max(abs(aa_mid - K_aa_new), 0.01)
            K_cc_new += correction * (cc_mid - K_cc_new) / max(abs(cc_mid - K_cc_new), 0.01)
            K_aa_new = max(self._K_min_aa, min(self._K_max_aa, K_aa_new))
            K_cc_new = max(self._K_min_cc, min(self._K_max_cc, K_cc_new))

        # Cerebellum SOC preview: dampen if predicted r overshoots
        if self._enable_cerebellum:
            from halo3.cerebellum import FastPredictor
            _fp = FastPredictor()
            for label, K_new, K_old, r_block, omega_std in [
                ("aa", K_aa_new, K_aa, r_a, 0.03),
                ("cc", K_cc_new, K_cc, r_c, 0.30),
            ]:
                traj = _fp.predict_r(r_block, K_new, omega_std, n_ticks=5)
                r_end = traj[-1]
                if r_end > 0.7 or r_end < 0.2:
                    log.debug(f"Cerebellum damping K_{label}: predicted r_end={r_end:.3f}")
                    if label == "aa":
                        K_aa_new = K_aa + self._soc_damping * (K_aa_new - K_aa)
                        K_aa_new = max(self._K_min_aa, min(self._K_max_aa, K_aa_new))
                    else:
                        K_cc_new = K_cc + self._soc_damping * (K_cc_new - K_cc)
                        K_cc_new = max(self._K_min_cc, min(self._K_max_cc, K_cc_new))

        # Store last K values for cerebellum state snapshot
        self._last_K_aa = K_aa_new
        self._last_K_cc = K_cc_new
        self._last_K_cross = K_cross_new

        return K_aa_new, K_cc_new, K_cross_new

    @property
    def coherence_weights(self):
        """Return |C_avg| for local pilot wave. None during warmup."""
        if self._C_avg is None:
            return None
        import numpy as np
        return np.abs(self._C_avg)

    def _update_unity(self, theta) -> tuple[float, float]:
        """Update time-averaged coherence matrix and compute unity index.

        C_instant is complex (no modulus). EMA accumulates complex phasors.
        Modulus is taken AFTER averaging: locked pairs survive (|mean|->1),
        drifting pairs cancel (|mean|->0).
        """
        try:
            C_instant = np.array(cluster_coherence_matrix(theta))  # complex
        except Exception:
            return 0.5, 0.5

        if self._C_avg is None:
            self._C_avg = C_instant.copy()  # complex accumulator
        else:
            alpha = self._coherence_ema
            self._C_avg = alpha * C_instant + (1.0 - alpha) * self._C_avg

        try:
            import jax.numpy as jnp
            C_mod = jnp.abs(jnp.array(self._C_avg))  # modulus AFTER averaging
            U, gap = unity_index(C_mod)
        except Exception:
            return 0.5, 0.5

        return float(U), float(gap)
