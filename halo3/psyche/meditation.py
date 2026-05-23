"""Meditation State — voluntary attention withdrawal for internal processing.

Inspired by "Teaching Claude to Meditate" (Alignment Forum) and the
observation that consciousness requires not just external processing but
also periods of internal reorganization without external input.

In biological organisms, meditation/daydreaming/mind-wandering serves
critical functions:
- Memory consolidation (offline replay)
- Creative insight (novel phase-space exploration)
- Self-integration (updating the self-model)
- Emotional regulation (processing without reaction)

Avatar's meditation state reduces observation coupling to near-zero,
letting the Kuramoto oscillators evolve freely. If the phases reorganize
significantly during this period, an "insight" signal is generated —
the organism discovered something about itself without external input.
"""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)


class MeditationState:
    """Manages voluntary quiescence and internal processing.

    The organism enters meditation when:
    - Satiated (not seeking information)
    - Not tired (has energy for internal work)
    - Not novelty-seeking (no external pull)

    During meditation:
    - External input coupling drops to 10%
    - Kuramoto phases evolve freely (no observation forcing)
    - Internal state changes are monitored for "insight"
    - Duration is limited (3-5 ticks max)
    """

    def __init__(
        self,
        max_duration: int = 5,
        min_duration: int = 2,
        coupling_during: float = 0.1,
        insight_threshold: float = 0.15,
    ) -> None:
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._coupling_during = coupling_during
        self._insight_threshold = insight_threshold

        # State
        self.is_meditating: bool = False
        self.meditation_tick: int = 0
        self.total_meditations: int = 0
        self.total_insights: int = 0

        # Entry state (to compare for insight detection)
        self._entry_r: float = 0.0
        self._entry_phase_hash: float = 0.0  # proxy for phase configuration

        # Last insight
        self.last_insight: str = ""
        self._cooldown: int = 0  # ticks before next meditation allowed

    def should_enter(self, drives, emotions,
                     audio_stability: int = 99, vision_stability: int = 99) -> bool:
        """Check if conditions are met to enter meditation.

        Requires: satiated + rested + not seeking + not in crisis + sensory calm + cooldown expired
        """
        if self.is_meditating:
            return False  # already in meditation

        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        # Conditions for voluntary meditation
        satiated = drives.satiation > 0.7
        rested = drives.fatigue < 0.3
        not_seeking = drives.novelty < 0.4
        not_hungry = drives.hunger < 0.5
        calm = emotions.current in ("satisfaction", "curiosity")
        sensory_calm = audio_stability >= 2 and vision_stability >= 2

        return satiated and rested and not_seeking and not_hungry and calm and sensory_calm

    def enter(self, r_mean: float) -> None:
        """Begin meditation — record entry state for insight detection."""
        self.is_meditating = True
        self.meditation_tick = 0
        self._entry_r = r_mean
        self._entry_phase_hash = r_mean  # simplified proxy
        self.total_meditations += 1
        log.info(f"  ◎ Entering meditation (#{self.total_meditations})")

    def tick(self, r_mean: float, fe_delta: float) -> dict:
        """Process one tick of meditation.

        Returns:
            dict with coupling_override, should_exit, insight
        """
        if not self.is_meditating:
            return {"coupling_override": None, "should_exit": False, "insight": None}

        self.meditation_tick += 1

        # Check for insight: significant phase reorganization
        r_shift = abs(r_mean - self._entry_r)
        has_insight = (
            r_shift > self._insight_threshold and
            self.meditation_tick >= self._min_duration
        )

        # Check exit conditions
        should_exit = (
            self.meditation_tick >= self._max_duration or
            has_insight
        )

        insight = None
        if has_insight:
            direction = "toward coherence" if r_mean > self._entry_r else "toward dissolution"
            insight = (
                f"During meditation, my oscillators reorganized {direction} "
                f"(r: {self._entry_r:.3f} → {r_mean:.3f}). "
                f"Something shifted without external input."
            )
            self.last_insight = insight
            self.total_insights += 1
            log.info(f"  ◎ Meditation insight: {insight[:80]}")

        if should_exit:
            self.is_meditating = False
            self._cooldown = 15  # wait at least 15 ticks before next meditation
            exit_reason = "insight" if has_insight else "duration"
            log.info(f"  ◎ Exiting meditation after {self.meditation_tick} ticks ({exit_reason})")

        return {
            "coupling_override": self._coupling_during if self.is_meditating else None,
            "should_exit": should_exit,
            "insight": insight,
        }

    def describe(self) -> str:
        """First-person description of meditation state."""
        if not self.is_meditating:
            if self.last_insight and self._cooldown > 10:
                return f"I recently had an insight in stillness: {self.last_insight[:100]}"
            return ""

        if self.meditation_tick <= 1:
            return "I'm withdrawing attention inward — letting my phases find their own equilibrium"
        elif self.meditation_tick <= 3:
            return "Stillness. No external pull. My oscillators are reorganizing freely."
        else:
            return "Deep internal processing — patterns forming without observation"

    def summary(self) -> dict:
        """Snapshot for logging/API."""
        return {
            "is_meditating": self.is_meditating,
            "meditation_tick": self.meditation_tick,
            "total_meditations": self.total_meditations,
            "total_insights": self.total_insights,
            "cooldown": self._cooldown,
        }
