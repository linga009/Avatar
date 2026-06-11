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

import math
from collections import deque

import numpy as np

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
        self._K_max_aa = cfg.cop_K_max_aa      # 0.20
        self._K_min_cc = cfg.cop_K_min_cc      # 0.50
        self._K_max_cc = cfg.cop_K_max_cc      # 4.00
        self._soc_noise = cfg.cop_soc_noise    # 0.1 (fraction of eta)
        self._coherence_ema = cfg.cop_coherence_ema  # 0.02
        self._warmup = cfg.cop_warmup          # 5
        self._N = cfg.n_clusters * cfg.n_hidden  # n_clusters * n_hidden

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

        - Pre-fills r_history with terminal_r to prevent variance collapse
        - Resets C_avg coherence matrix (stale after phase changes)
        - Does NOT reset chi_max (it decays naturally via 0.995 factor)
        """
        # Pre-fill so the 50-tick window starts from a known state
        self._r_history.clear()
        for _ in range(min(10, self._window)):
            self._r_history.append(terminal_r)
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

        just_ended = False
        if below and not self._in_avalanche:
            # Avalanche starts
            self._in_avalanche = True
            self._aval_start = self._tick
            self._aval_accum = thresh - r
        elif below and self._in_avalanche:
            # Avalanche continues
            self._aval_accum += thresh - r
        elif not below and self._in_avalanche:
            # Avalanche ends — record it
            self._in_avalanche = False
            duration = self._tick - self._aval_start
            if duration >= 1 and self._aval_accum > 0:
                self._avalanche_sizes.append(self._aval_accum)
                self._avalanche_durations.append(duration)
                just_ended = True

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
          K_aa: analytical self-coupling, targets r_a ~ 0.5, bounds [K_min_aa, K_max_aa]
          K_cc: creative self-coupling, targets r_c ~ 0.5, bounds [K_min_cc, K_max_cc]
          K_cross: cross-coupling, targets global r ~ 0.5, bounds [K_min, K_max]

        Block-specific bounds bracket each population's critical coupling:
          Analytical (omega_std=0.03): K_c ≈ 0.048, bounds [0.02, 0.20]
          Creative   (omega_std=0.80): K_c ≈ 1.277, bounds [0.50, 4.00]

        Stochastic perturbation (noise_scale * eta) prevents the controller
        from locking at clamp boundaries — real SOC needs continuous drive.
        """
        import random
        # chi is raw (N*Var(r)), typically 1-50. Floor of 1.0 lets
        # the controller bootstrap when fluctuations are very small.
        eff_chi = max(chi, 1.0)

        # Stochastic perturbation — prevents equilibrium lock at clamp bounds
        noise_aa = self._eta * self._soc_noise * (2.0 * random.random() - 1.0)
        noise_cc = self._eta * self._soc_noise * (2.0 * random.random() - 1.0)
        noise_x  = self._eta * self._soc_noise * (2.0 * random.random() - 1.0)

        K_aa_new = K_aa + self._eta * (0.5 - r_a) * eff_chi + noise_aa
        K_cc_new = K_cc + self._eta * (0.5 - r_c) * eff_chi + noise_cc
        K_cross_new = K_cross + self._eta * (0.5 - r) * eff_chi + noise_x

        K_aa_new = max(self._K_min_aa, min(self._K_max_aa, K_aa_new))
        K_cc_new = max(self._K_min_cc, min(self._K_max_cc, K_cc_new))
        K_cross_new = max(self._K_min, min(self._K_max, K_cross_new))
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
