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

        self._r_history: deque[float] = deque(maxlen=self._window)
        self._fe_history: deque[float] = deque(maxlen=self._window)

        self._chi_max: float = 1.0
        self._C_avg: np.ndarray | None = None
        self._tick: int = 0

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

        chi = self._compute_chi()
        tau = self._compute_tau()
        T_body = abs(r_a - r_c)
        f_dot = -fe_delta
        unity_val, gap = self._update_unity(theta)

        if self._tick <= self._warmup:
            K_new = K
        else:
            K_new = self._soc_update(K, r_mean, chi)

        U_product = r_mean * chi

        return {
            "chi": chi,
            "tau": tau,
            "unity": unity_val,
            "gap": gap,
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

    def _soc_update(self, K: float, r: float, chi: float) -> float:
        """Self-organized criticality controller.

        Drives system toward K* where U = r * chi is maximal.

        Uses max(chi, 0.1) as effective chi so the controller can
        bootstrap from far-from-critical states where chi ~ 0.
        Near criticality chi >> 0.1, so the floor has no effect.
        """
        r_error = 0.5 - r
        effective_chi = max(chi, 0.1)  # floor prevents frozen K when subcritical
        K_dot = self._eta * r_error * effective_chi
        new_K = K + K_dot
        return max(self._K_min, min(self._K_max, new_K))

    def _update_unity(self, theta) -> tuple[float, float]:
        """Update time-averaged coherence matrix and compute unity index."""
        try:
            C_instant = np.array(cluster_coherence_matrix(theta))
        except Exception:
            return 0.5, 0.5

        if self._C_avg is None:
            self._C_avg = C_instant.copy()
        else:
            alpha = self._coherence_ema
            self._C_avg = alpha * C_instant + (1.0 - alpha) * self._C_avg

        try:
            import jax.numpy as jnp
            U, gap = unity_index(jnp.array(self._C_avg))
        except Exception:
            return 0.5, 0.5

        return float(U), float(gap)
