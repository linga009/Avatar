"""Drives — the organism's primary needs that motivate behavior.

Inspired by Panksepp's affective neuroscience:
- Hunger: need for information (new free energy reduction)
- Fatigue: accumulated processing cost (need for dreaming)
- Curiosity: attraction to novelty at the edge of understanding
- Satiation: restlessness from sustained high sync on same topic
- Starvation: emergency state from sustained zero information input
- Novelty: need for fundamentally different topics (distinct from curiosity)

v3.1 additions:
  - Starvation drive: separate from hunger, triggers emergency override
  - Novelty drive: monotonically increases on same topic cluster
  - Exploration budget: tracks explore/exploit ratio
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
    starvation: float = 0.0    # [0,1] emergency: zero information input
    novelty: float = 0.0       # [0,1] need for a fundamentally different topic
    ticks_since_learning: int = 0
    ticks_high_r: int = 0
    ticks_zero_input: int = 0  # consecutive ticks with no search results
    _explore_count: int = 0    # topic changes
    _exploit_count: int = 0    # same-topic ticks

    def update(
        self,
        r_mean: float,
        fe_delta: float,
        perception_failed: bool = False,
        topic_changed: bool = False,
        dt: float = 1.0,
        sensory_arousal: float = 0.0,
        sensory_novelty: float = 0.0,
        chi_norm: float = 0.5,
        graph_metrics: dict | None = None,
    ) -> None:
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
        fatigue_rate += 0.002 * sensory_arousal  # sensory load increases tiredness
        self.fatigue = min(1.0, self.fatigue + fatigue_rate * dt)

        # --- Satiation (COP: ordered + rigid = nothing new to learn) ---
        if r_mean > 0.55 and chi_norm < 0.2:
            self.ticks_high_r += 1
            self.satiation = min(1.0, self.satiation + 0.08)
        else:
            self.ticks_high_r = 0
            self.satiation = max(0.0, self.satiation - 0.1)

        # Graph topology modulates satiation and curiosity
        if graph_metrics:
            # Dense local graph = well-understood → satiate faster
            if graph_metrics.get("avg_clustering", 0) > 0.7:
                self.satiation = min(1.0, self.satiation + 0.04)
            # Many frontier nodes = more to explore → boost curiosity
            frontier_ratio = graph_metrics.get("frontier_ratio", 0.5)
            if frontier_ratio > 0.3:
                self.curiosity = min(1.0, self.curiosity + 0.05 * frontier_ratio)

        # --- Starvation (emergency: no information flowing in) ---
        if perception_failed:
            self.ticks_zero_input += 1
            self.starvation = min(1.0, self.starvation + 0.15)
        else:
            self.ticks_zero_input = 0
            self.starvation = max(0.0, self.starvation - 0.3)

        # Senses confirm world exists — dampens starvation
        if sensory_arousal > 0.3:
            self.starvation = max(0.0, self.starvation - 0.1)

        # --- Novelty (need for fundamentally different topics) ---
        if topic_changed:
            self.novelty = max(0.0, self.novelty - 0.4)
            self._explore_count += 1
        else:
            self.novelty = min(1.0, self.novelty + 0.02)
            self._exploit_count += 1

        # --- Curiosity (COP: chi IS curiosity) ---
        self.curiosity = min(1.0, chi_norm + self.starvation * 0.3)

        # High sensory novelty pulls toward exploration
        if sensory_novelty > 0.7:
            self.curiosity = min(1.0, self.curiosity + 0.03 * sensory_novelty)

    def dream_reset(self) -> None:
        self.fatigue = 0.1
        self.hunger = max(0.3, self.hunger * 0.5)
        self.starvation = 0.0
        self.ticks_zero_input = 0

    @property
    def needs_dream(self) -> bool:
        return self.fatigue > 0.65

    @property
    def is_starving(self) -> bool:
        """True when information hunger is desperate."""
        return self.hunger > 0.85

    @property
    def is_information_starved(self) -> bool:
        """True when zero input has persisted — emergency override needed."""
        return self.starvation > 0.5 or self.ticks_zero_input >= 3

    @property
    def is_satiated(self) -> bool:
        return self.satiation > 0.6

    @property
    def needs_novelty(self) -> bool:
        """True when the organism has been on the same topic cluster too long."""
        return self.novelty > 0.7

    @property
    def exploration_ratio(self) -> float:
        """Fraction of ticks that were exploration (topic changes)."""
        total = self._explore_count + self._exploit_count
        if total == 0:
            return 0.5
        return self._explore_count / total

    def summary(self) -> str:
        h_bar = "█" * int(self.hunger * 10) + "░" * (10 - int(self.hunger * 10))
        f_bar = "█" * int(self.fatigue * 10) + "░" * (10 - int(self.fatigue * 10))
        c_bar = "█" * int(self.curiosity * 10) + "░" * (10 - int(self.curiosity * 10))
        base = f"hunger=[{h_bar}] fatigue=[{f_bar}] curiosity=[{c_bar}]"
        if self.satiation > 0.1:
            s_bar = "█" * int(self.satiation * 10) + "░" * (10 - int(self.satiation * 10))
            base += f" satiation=[{s_bar}]"
        if self.starvation > 0.1:
            st_bar = "█" * int(self.starvation * 10) + "░" * (10 - int(self.starvation * 10))
            base += f" STARVING=[{st_bar}]"
        return base
