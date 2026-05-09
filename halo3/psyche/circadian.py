"""Circadian Rhythm — biological-like cycles of alertness and rest.

Modulates the physics engine's behavior based on time of day
and accumulated fatigue. Creates genuine need for dreaming.
"""
from __future__ import annotations
import datetime
import math


class CircadianClock:
    """Tracks biological-like cycles that modulate the organism."""

    def __init__(self, dream_hour: int = 2, dream_duration_min: int = 15) -> None:
        self.dream_hour = dream_hour
        self.dream_duration_min = dream_duration_min
        self._last_dream_date: str | None = None

    @property
    def alertness(self) -> float:
        """Current alertness based on time of day. Peaks at noon, troughs at 3am.

        Returns float in [0.3, 1.0].
        """
        hour = datetime.datetime.now().hour + datetime.datetime.now().minute / 60.0
        # Cosine curve: peak at 14:00, trough at 02:00
        phase = (hour - 14.0) / 24.0 * 2 * math.pi
        raw = 0.5 + 0.5 * math.cos(phase)  # [0, 1]
        return 0.3 + 0.7 * raw  # [0.3, 1.0]

    @property
    def is_dream_window(self) -> bool:
        """True during the nightly dream window."""
        now = datetime.datetime.now()
        return now.hour == self.dream_hour and now.minute < self.dream_duration_min

    @property
    def should_dream_today(self) -> bool:
        """True if we haven't dreamed today yet and we're in the window."""
        today = datetime.date.today().isoformat()
        return self.is_dream_window and self._last_dream_date != today

    def mark_dreamed(self) -> None:
        """Record that dreaming happened today."""
        self._last_dream_date = datetime.date.today().isoformat()

    def modulate_coupling(self, base_K: float, fatigue: float) -> float:
        """Modulate Kuramoto coupling based on alertness and fatigue.

        High alertness + low fatigue → strong coupling (focused)
        Low alertness + high fatigue → weak coupling (diffuse)
        """
        effective = self.alertness * (1.0 - fatigue * 0.5)
        return base_K * effective

    def modulate_tick_interval(self, base_interval: float, fatigue: float) -> float:
        """Tired organisms think slower."""
        # Up to 2x slower when exhausted
        slowdown = 1.0 + fatigue * 1.0
        return base_interval * slowdown
