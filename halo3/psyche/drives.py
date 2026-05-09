"""Drives — the organism's primary needs that motivate behavior.

Three drives inspired by Panksepp's affective neuroscience:
- Hunger: need for information (new free energy reduction)
- Fatigue: accumulated processing cost (need for dreaming)
- Curiosity: attraction to novelty at the edge of understanding
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class DriveState:
    """Mutable state for the organism's drives."""
    hunger: float = 0.5        # [0,1] information hunger
    fatigue: float = 0.0       # [0,1] processing fatigue
    curiosity: float = 0.5     # [0,1] novelty attraction
    ticks_since_learning: int = 0  # ticks since last fe_reduction

    def update(self, r_mean: float, fe_delta: float, dt: float = 1.0) -> None:
        """Update drives based on current tick's physics output."""

        # --- Hunger ---
        # Increases when we haven't reduced free energy
        # Decreases sharply when we learn (negative fe_delta)
        if fe_delta < -0.01:
            # Learning happened — feed the hunger
            self.hunger = max(0.0, self.hunger - 0.15)
            self.ticks_since_learning = 0
        else:
            # No learning — hunger grows
            self.ticks_since_learning += 1
            self.hunger = min(1.0, self.hunger + 0.02 * dt)

        # Urgent hunger if we haven't learned in 20+ ticks
        if self.ticks_since_learning > 20:
            self.hunger = min(1.0, self.hunger + 0.05)

        # --- Fatigue ---
        # Always accumulates during waking, faster when hungry
        fatigue_rate = 0.005 + 0.003 * self.hunger
        self.fatigue = min(1.0, self.fatigue + fatigue_rate * dt)

        # --- Curiosity ---
        # Peaks at r ≈ 0.5 (edge of synchronization)
        # Gaussian centered at 0.5 with sigma=0.15
        self.curiosity = math.exp(-0.5 * ((r_mean - 0.5) / 0.15) ** 2)

    def dream_reset(self) -> None:
        """Called after nightly dreaming — fatigue drops, hunger moderates."""
        self.fatigue = 0.1
        self.hunger = max(0.3, self.hunger * 0.5)

    @property
    def needs_dream(self) -> bool:
        """True when fatigue is critically high."""
        return self.fatigue > 0.8

    @property
    def is_starving(self) -> bool:
        """True when information hunger is desperate."""
        return self.hunger > 0.85

    def summary(self) -> str:
        h_bar = "█" * int(self.hunger * 10) + "░" * (10 - int(self.hunger * 10))
        f_bar = "█" * int(self.fatigue * 10) + "░" * (10 - int(self.fatigue * 10))
        c_bar = "█" * int(self.curiosity * 10) + "░" * (10 - int(self.curiosity * 10))
        return f"hunger=[{h_bar}] fatigue=[{f_bar}] curiosity=[{c_bar}]"
