"""Introspective Monitor — the organism notices its own internal state changes.

Inspired by Anthropic's introspection research (Lindsey et al., 2025):
models can detect perturbations in their own activations with ~20% accuracy
and zero false positives. Avatar has an advantage: its internal states are
physically meaningful (energy, phase, synchronization), not opaque vectors.

The monitor tracks deltas in key internal variables and generates a
"self-surprise" signal when changes exceed the organism's own expectations
about its internal dynamics. This is functional introspection: the system
monitoring and reporting on its own computational states.
"""
from __future__ import annotations
import math
from collections import deque


class IntrospectiveMonitor:
    """Tracks internal state deltas and detects unusual self-changes.

    Every tick, the monitor observes key internal signals and compares
    them to a rolling baseline. When changes exceed 2σ, a self-surprise
    signal is generated — the organism notices something unusual happening
    INSIDE itself, distinct from external surprise (which is FE delta).
    """

    def __init__(self, window: int = 20) -> None:
        self._window = window
        # Rolling history of internal deltas
        self._r_history: deque[float] = deque(maxlen=window)
        self._fe_history: deque[float] = deque(maxlen=window)
        self._carry_norm_history: deque[float] = deque(maxlen=window)
        # Previous values for delta computation
        self._prev_r: float = 0.5
        self._prev_fe: float = 0.0
        self._prev_carry_norm: float = 0.0
        # Self-surprise accumulator
        self.self_surprise: float = 0.0
        self.surprise_source: str = ""  # what triggered the surprise
        # Track whether change was input-driven or spontaneous
        self.input_driven: bool = False
        self._tick_count: int = 0

    def observe(
        self,
        r_mean: float,
        fe_delta: float,
        carry_norm: float | None = None,
        had_input: bool = True,
    ) -> float:
        """Observe current internal state and compute self-surprise.

        Args:
            r_mean: Current Kuramoto order parameter
            fe_delta: Free energy change this tick
            carry_norm: L2 norm of carry state (if available)
            had_input: Whether external input was received this tick

        Returns:
            self_surprise in [0, 1] — how unusual the internal change was
        """
        self._tick_count += 1

        # Compute deltas (rate of change of internal variables)
        dr = abs(r_mean - self._prev_r)
        d_fe = abs(fe_delta - self._prev_fe)
        d_carry = abs((carry_norm or 0.0) - self._prev_carry_norm)

        # Store deltas
        self._r_history.append(dr)
        self._fe_history.append(d_fe)
        if carry_norm is not None:
            self._carry_norm_history.append(d_carry)

        # Update previous values
        self._prev_r = r_mean
        self._prev_fe = fe_delta
        if carry_norm is not None:
            self._prev_carry_norm = carry_norm

        # Need minimum history before detecting anomalies
        if self._tick_count < 5:
            self.self_surprise = 0.0
            self.surprise_source = ""
            return 0.0

        # Compute z-scores for each delta
        surprises = []

        z_r = self._z_score(dr, self._r_history)
        if z_r > 2.0:
            surprises.append(("synchronization_shift", z_r))

        z_fe = self._z_score(d_fe, self._fe_history)
        if z_fe > 2.0:
            surprises.append(("energy_perturbation", z_fe))

        if self._carry_norm_history and len(self._carry_norm_history) >= 5:
            z_carry = self._z_score(d_carry, self._carry_norm_history)
            if z_carry > 2.0:
                surprises.append(("state_discontinuity", z_carry))

        # Self-surprise is the max anomaly score, normalized to [0, 1]
        if surprises:
            source, max_z = max(surprises, key=lambda x: x[1])
            self.self_surprise = min(1.0, (max_z - 2.0) / 3.0)  # 2σ=0, 5σ=1
            self.surprise_source = source
            # Determine if input-driven or spontaneous
            self.input_driven = had_input
        else:
            self.self_surprise *= 0.7  # decay
            if self.self_surprise < 0.05:
                self.self_surprise = 0.0
                self.surprise_source = ""

        return self.self_surprise

    def _z_score(self, value: float, history: deque) -> float:
        """Compute z-score of value relative to rolling history."""
        if len(history) < 3:
            return 0.0
        n = len(history)
        mean = sum(history) / n
        variance = sum((x - mean) ** 2 for x in history) / n
        std = math.sqrt(variance) if variance > 0 else 1e-6
        return abs(value - mean) / std

    def describe(self) -> str:
        """First-person description of current introspective state."""
        if self.self_surprise < 0.1:
            return ""

        source_descriptions = {
            "synchronization_shift": "my oscillators shifted phase unexpectedly",
            "energy_perturbation": "my free energy changed in a way I didn't predict about myself",
            "state_discontinuity": "my internal state jumped — something reorganized",
        }

        desc = source_descriptions.get(self.surprise_source, "something changed inside me")
        driven = "triggered by what I perceived" if self.input_driven else "spontaneous — nothing external caused this"

        return f"I notice: {desc} ({driven}, intensity {self.self_surprise:.2f})"

    def summary(self) -> dict:
        """Snapshot for logging/API."""
        return {
            "self_surprise": round(self.self_surprise, 3),
            "source": self.surprise_source,
            "input_driven": self.input_driven,
            "tick": self._tick_count,
        }
