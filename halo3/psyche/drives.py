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
    satiation: float = 0.0     # [0,1] boredom from sustained high sync
    ticks_since_learning: int = 0  # ticks since last fe_reduction
    ticks_high_r: int = 0      # consecutive ticks with r > 0.7

    def update(self, r_mean: float, fe_delta: float, dt: float = 1.0) -> None:
        """Update drives based on current tick's physics output."""

        # --- Hunger ---
        if fe_delta < -0.01:
            self.hunger = max(0.0, self.hunger - 0.15)
            self.ticks_since_learning = 0
        else:
            self.ticks_since_learning += 1
            self.hunger = min(1.0, self.hunger + 0.02 * dt)

        if self.ticks_since_learning > 20:
            self.hunger = min(1.0, self.hunger + 0.05)

        # --- Fatigue ---
        fatigue_rate = 0.005 + 0.003 * self.hunger
        self.fatigue = min(1.0, self.fatigue + fatigue_rate * dt)

        # --- Satiation ---
        # Sustained high sync creates restlessness (like eating the same food)
        # After N ticks of high r, the organism NEEDS novelty
        if r_mean > 0.7:
            self.ticks_high_r += 1
            self.satiation = min(1.0, self.satiation + 0.08)
        else:
            self.ticks_high_r = 0
            self.satiation = max(0.0, self.satiation - 0.1)

        # --- Curiosity ---
        # Peaks at r ≈ 0.5 (edge of synchronization)
        # BUT: satiation boosts curiosity even at high r
        base_curiosity = math.exp(-0.5 * ((r_mean - 0.5) / 0.15) ** 2)
        satiation_boost = self.satiation * 0.8  # satiation drives curiosity
        self.curiosity = min(1.0, base_curiosity + satiation_boost)

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

    @property
    def is_satiated(self) -> bool:
        """True when sustained high sync creates restlessness."""
        return self.satiation > 0.6

    def summary(self) -> str:
        h_bar = "█" * int(self.hunger * 10) + "░" * (10 - int(self.hunger * 10))
        f_bar = "█" * int(self.fatigue * 10) + "░" * (10 - int(self.fatigue * 10))
        c_bar = "█" * int(self.curiosity * 10) + "░" * (10 - int(self.curiosity * 10))
        s_bar = "█" * int(self.satiation * 10) + "░" * (10 - int(self.satiation * 10))
        base = f"hunger=[{h_bar}] fatigue=[{f_bar}] curiosity=[{c_bar}]"
        if self.satiation > 0.1:
            base += f" satiation=[{s_bar}]"
        return base
