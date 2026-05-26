"""Introspective Monitor — self-surprise from relaxation time dynamics.

COP v4.0: Rising tau (critical slowing) = "something is building up."
Sudden tau drop = "it just reorganized." Self-surprise is the magnitude
of unexpected tau changes — the dynamical signature of impending or
just-completed phase transitions.
"""
from __future__ import annotations
import math
from collections import deque


class IntrospectiveMonitor:
    """Tracks tau dynamics and detects unusual internal state changes."""

    def __init__(self, window: int = 20) -> None:
        self._window = window
        self._tau_history: deque[float] = deque(maxlen=window)
        self.self_surprise: float = 0.0
        self.surprise_source: str = ""
        self.input_driven: bool = False
        self._tick_count: int = 0

    def observe(
        self,
        r_mean: float,
        fe_delta: float,
        carry_norm: float | None = None,
        had_input: bool = True,
        tau: float = 0.5,
    ) -> float:
        """Observe current state and compute self-surprise from tau.

        Args:
            r_mean: order parameter (kept for interface compat)
            fe_delta: free energy delta (kept for interface compat)
            carry_norm: carry state norm (kept for interface compat)
            had_input: whether external input was received
            tau: relaxation time from COP engine

        Returns:
            self_surprise in [0, 1]
        """
        self._tick_count += 1
        self._tau_history.append(tau)

        if self._tick_count < 5:
            self.self_surprise = 0.0
            self.surprise_source = ""
            return 0.0

        # Compute tau derivative (rate of change)
        if len(self._tau_history) >= 2:
            d_tau = abs(self._tau_history[-1] - self._tau_history[-2])
        else:
            d_tau = 0.0

        # z-score of tau derivative against recent history
        tau_deltas = [abs(self._tau_history[i] - self._tau_history[i - 1])
                      for i in range(1, len(self._tau_history))]
        if len(tau_deltas) < 3:
            self.self_surprise *= 0.7
            return self.self_surprise

        mean_d = sum(tau_deltas) / len(tau_deltas)
        var_d = sum((x - mean_d) ** 2 for x in tau_deltas) / len(tau_deltas)
        std_d = math.sqrt(var_d) if var_d > 0 else 1e-6
        z = abs(d_tau - mean_d) / std_d

        if z > 2.0:
            self.self_surprise = min(1.0, (z - 2.0) / 3.0)
            if len(self._tau_history) >= 2:
                if self._tau_history[-1] > self._tau_history[-2]:
                    self.surprise_source = "critical_slowing"
                else:
                    self.surprise_source = "phase_reorganization"
            self.input_driven = had_input
        else:
            self.self_surprise *= 0.7
            if self.self_surprise < 0.05:
                self.self_surprise = 0.0
                self.surprise_source = ""

        return self.self_surprise

    def describe(self) -> str:
        if self.self_surprise < 0.1:
            return ""
        source_descriptions = {
            "critical_slowing": "my relaxation time is stretching — something is building",
            "phase_reorganization": "my oscillators just reorganized — something shifted",
        }
        desc = source_descriptions.get(self.surprise_source, "something changed inside me")
        driven = "triggered by input" if self.input_driven else "spontaneous"
        return f"I notice: {desc} ({driven}, intensity {self.self_surprise:.2f})"

    def summary(self) -> dict:
        return {
            "self_surprise": round(self.self_surprise, 3),
            "source": self.surprise_source,
            "input_driven": self.input_driven,
            "tick": self._tick_count,
        }
