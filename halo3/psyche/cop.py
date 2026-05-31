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
        self._eta = cfg.cop_eta                # 0.005
        self._K_min = cfg.cop_K_min            # 0.05
        self._K_max = cfg.cop_K_max            # 2.0
        self._coherence_ema = cfg.cop_coherence_ema  # 0.02
        self._warmup = cfg.cop_warmup          # 5
        self._N = cfg.n_clusters * cfg.n_hidden  # n_clusters * n_hidden

        self._r_history: deque[float] = deque(maxlen=self._window)
        self._fe_history: deque[float] = deque(maxlen=self._window)
        self._obs_norm_history: deque[float] = deque(maxlen=self._window)

        self._chi_max: float = 1.0
        self._C_avg: np.ndarray | None = None
        self._tick: int = 0

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

        chi = self._compute_chi()
        tau = self._compute_tau()
        T_body = abs(r_a - r_c)
        f_dot = -fe_delta
        import jax.numpy as jnp
        _theta = theta if theta is not None else jnp.zeros((1, 1))
        unity_val, gap = self._update_unity(_theta)

        if self._tick <= self._warmup:
            K_aa_new, K_cc_new, K_cross_new = K_aa, K_cc, K_cross
        else:
            K_aa_new, K_cc_new, K_cross_new = self._soc_update(
                K_aa, K_cc, K_cross, r_mean, r_a, r_c, chi)

        U_product = r_mean * chi

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

            # Effective temperature from COP observables
            T_eff = chi * (1.0 + tau)

            F_thermo = float(H_mean) - T_eff * S_phase

        return {
            "chi": chi,
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
        }

    def _compute_chi(self) -> float:
        """Susceptibility with Harada-Sasa FDT-violation correction.

        Raw chi = N * Var(r). Harada-Sasa: compute lag-1 autocorrelation
        C(1) of r and lag-1 cross-correlation R(1) of (r, obs_norm).
        FDT violation sigma = max(0, C(1) - R(1)) measures entropy
        production. chi is downweighted when driving is strong.
        """
        if len(self._r_history) < 5:
            return 0.5

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

    def _soc_update(self, K_aa: float, K_cc: float, K_cross: float,
                    r: float, r_a: float, r_c: float, chi: float) -> tuple[float, float, float]:
        """Proportional criticality controller — block coupling.

        Three independent controllers:
          K_aa: analytical self-coupling, targets r_a ~ 0.5
          K_cc: creative self-coupling, targets r_c ~ 0.5
          K_cross: cross-coupling, targets global r ~ 0.5

        Drives each K toward r~0.5 where susceptibility peaks. Not true SOC
        in the BTW sandpile sense -- this is a feedback loop, not
        emergent criticality.

        Uses max(chi, 0.1) as effective chi so the controller can
        bootstrap from far-from-critical states where chi ~ 0.
        Near criticality chi >> 0.1, so the floor has no effect.
        """
        eff_chi = max(chi, 0.1)

        K_aa_new = K_aa + self._eta * (0.5 - r_a) * eff_chi
        K_cc_new = K_cc + self._eta * (0.5 - r_c) * eff_chi
        K_cross_new = K_cross + self._eta * (0.5 - r) * eff_chi

        clamp = lambda x: max(self._K_min, min(self._K_max, x))
        return clamp(K_aa_new), clamp(K_cc_new), clamp(K_cross_new)

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
