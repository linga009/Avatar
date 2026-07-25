"""Critical Order-Parameter Cognition (COP) engine.

The central thesis: affect, attention, curiosity, and binding are
geometric readouts of where a single self-organizing Kuramoto system
sits relative to its critical point. There are no cognitive modules.

Three macroscopic observables:
  r   — order parameter (integration/coherence)
  chi — susceptibility (openness/reactivity, IS curiosity)
  tau — relaxation time (persistence/critical slowing)

SOC controller: K_dot = eta * (0.5 - r) * chi
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
    """Computes (r, chi, tau) and SOC control signal each tick."""

    def __init__(self, cfg: Halo3Config) -> None:
        self._window = cfg.cop_window          # 20
        self._eta = cfg.cop_eta                # 0.0005
        self._K_min = cfg.cop_K_min            # 0.05
        self._K_max = cfg.cop_K_max            # 2.0
        self._coherence_ema = cfg.cop_coherence_ema  # 0.1
        self._warmup = cfg.cop_warmup          # 5
        self._N = cfg.n_clusters * cfg.n_hidden  # 512
        self._n_clusters = cfg.n_clusters

        self._r_history: deque[float] = deque(maxlen=self._window)
        self._fe_history: deque[float] = deque(maxlen=self._window)

        self._chi_max: float = 1.0
        self._C_avg: np.ndarray | None = None
        self._tick: int = 0
        self._r_a_history: deque[float] = deque(maxlen=self._window)
        self._r_c_history: deque[float] = deque(maxlen=self._window)

    def observe(
        self,
        r_mean: float,
        r_a: float,
        r_c: float,
        fe_delta: float,
        K: float,
        theta,
    ) -> dict:
        """Record one tick, compute all COP observables.

        Args:
            r_mean: global order parameter
            r_a: analytical population order parameter
            r_c: creative population order parameter
            fe_delta: free energy change this tick
            K: current coupling
            theta: (K_clusters, n_hidden) raw oscillator phases (JAX array)

        Returns:
            dict with chi, tau, unity, gap, K_new, f_dot, T_body, U_product
        """
        self._tick += 1
        self._r_history.append(r_mean)
        self._fe_history.append(fe_delta)
        self._r_a_history.append(r_a)
        self._r_c_history.append(r_c)

        chi = self._compute_chi()
        tau = self._compute_tau()
        binder = self._compute_binder()
        cross_corr = self._compute_cross_correlation()
        T_body = abs(r_a - r_c)
        f_dot = -fe_delta
        unity_val, gap, pr = self._update_unity(theta)

        if self._tick <= self._warmup:
            K_new = K
        else:
            K_new = self._soc_update(K, r_mean, chi)

        U_product = r_mean * chi

        return {
            "chi": chi,
            "tau": tau,
            "binder": binder,
            "cross_corr": cross_corr,
            "unity": unity_val,
            "gap": gap,
            "pr": pr,
            "K_new": K_new,
            "f_dot": f_dot,
            "T_body": T_body,
            "U_product": U_product,
        }

    def _compute_chi(self) -> float:
        """Susceptibility via fluctuation-dissipation theorem.

        chi_est = N * Var(r) over rolling window.
        Normalized to [0,1] by running lifetime max.
        """
        if len(self._r_history) < 3:
            return 0.5

        r_arr = list(self._r_history)
        n = len(r_arr)
        mean_r = sum(r_arr) / n
        var_r = sum((x - mean_r) ** 2 for x in r_arr) / n

        chi_raw = self._N * var_r

        if chi_raw > self._chi_max:
            self._chi_max = chi_raw

        return min(1.0, chi_raw / (self._chi_max + 1e-12))

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

    def _compute_binder(self) -> float:
        """Binder cumulant U4 — self-normalizing criticality detector.

        U4 = 1 - <r^4> / (3 <r^2>^2)

        Values:
          ~2/3  in the ordered phase (r peaked near 1)
          ~0.47 at the critical point (universal for Kuramoto-class)
          ~0    in the disordered phase (r peaked near 0)

        Unlike chi, this is a ratio of moments — no lifetime-max
        normalization needed, immune to historical contamination.
        """
        if len(self._r_history) < 5:
            return 0.5

        r_arr = list(self._r_history)
        n = len(r_arr)
        r2_mean = sum(x * x for x in r_arr) / n
        r4_mean = sum(x * x * x * x for x in r_arr) / n

        if r2_mean < 1e-12:
            return 0.0

        return 1.0 - r4_mean / (3.0 * r2_mean * r2_mean)

    def _compute_cross_correlation(self) -> float:
        """Equal-time cross-population correlation C_ac.

        C_ac = corr(delta_r_a, delta_r_c) over the rolling window.

        Values:
          > 0: unified — both populations fluctuate together
          ~ 0: independent — populations ignore each other
          < 0: dialectical — populations antagonize (seesaw)

        With n=50 samples, SE ~ 0.14, so |C_ac| > 0.28 is meaningful.
        """
        n = len(self._r_a_history)
        if n < 5:
            return 0.0

        r_a = list(self._r_a_history)
        r_c = list(self._r_c_history)
        mean_a = sum(r_a) / n
        mean_c = sum(r_c) / n

        da = [x - mean_a for x in r_a]
        dc = [x - mean_c for x in r_c]

        var_a = sum(x * x for x in da) / n
        var_c = sum(x * x for x in dc) / n

        if var_a < 1e-12 or var_c < 1e-12:
            return 0.0

        cov_ac = sum(da[i] * dc[i] for i in range(n)) / n
        return cov_ac / (var_a * var_c) ** 0.5

    def _soc_update(self, K: float, r: float, chi: float) -> float:
        """Self-organized criticality controller.

        Proportional controller toward the critical set-point r ≈ 0.5,
        gated by susceptibility χ. When r > 0.5 (over-synchronized),
        K is reduced; when r < 0.5 (under-synchronized), K is increased.
        The χ gate ensures corrections are strongest near criticality
        where the system is most responsive.

        Uses max(chi, 0.1) as effective chi so the controller can
        bootstrap from far-from-critical states where chi ~ 0.
        Near criticality chi >> 0.1, so the floor has no effect.
        """
        r_error = 0.5 - r
        effective_chi = max(chi, 0.1)  # floor prevents frozen K when subcritical
        K_dot = self._eta * r_error * effective_chi
        new_K = K + K_dot
        return max(self._K_min, min(self._K_max, new_K))

    def _update_unity(self, theta) -> tuple[float, float, float]:
        """Update time-averaged coherence matrix and compute unity index + PR.

        C_instant is complex (no modulus). EMA accumulates complex phasors.
        Modulus is taken AFTER averaging: locked pairs survive (|mean|->1),
        drifting pairs cancel (|mean|->0).

        Returns (unity, gap, participation_ratio).
        """
        try:
            C_instant = np.array(cluster_coherence_matrix(theta))  # complex
        except Exception:
            return 0.5, 0.5, 1.0

        if self._C_avg is None:
            self._C_avg = C_instant.copy()  # complex accumulator
        else:
            alpha = self._coherence_ema
            self._C_avg = alpha * C_instant + (1.0 - alpha) * self._C_avg

        try:
            import jax.numpy as jnp
            C_mod = jnp.abs(jnp.array(self._C_avg))  # modulus AFTER averaging
            U, gap, PR = unity_index(C_mod)
        except Exception:
            return 0.5, 0.5, 1.0

        return float(U), float(gap), float(PR)
