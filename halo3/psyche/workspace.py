"""Global Workspace — all-or-none ignition and broadcast.

Implements Baars/Dehaene's Global Workspace Theory (GWT):
- Many modular processors compete for access to a limited-capacity workspace
- When synchronization exceeds threshold: IGNITION (all-or-none)
- Winning content is broadcast globally to all modules
- Below threshold: processing continues locally (unconscious)

In Avatar, Kuramoto synchronization IS the competition.
High r = ignition = the organism becomes CONSCIOUS of the pattern.
Low r = local processing = unconscious computation continues.

The broadcast vector represents WHAT the organism is conscious of
at this moment — the content of its experience.

Ignition threshold anchored to SOC critical point (r=0.5).
Unity (eigenvalue dominance) scales broadcast intensity — vivid
vs dim consciousness. Chi at the moment of crossing records
transition sharpness — dramatic vs quiet ignition.
"""
from __future__ import annotations
from collections import deque


class GlobalWorkspace:
    """Implements GWT ignition and broadcast for Avatar.

    The workspace has two states:
    - DARK: effective_r < ignition_threshold (0.5). Processing is
      local/unconscious. The organism computes but is not "aware."
    - IGNITED: effective_r >= ignition_threshold. The dominant pattern
      is broadcast to all modules. The organism is CONSCIOUS.

    effective_r = r_mean + 0.05 * sensory_novelty (attention capture).

    Hysteresis prevents flickering: once ignited, stays ignited until
    effective_r drops below sustain_threshold (0.4).

    Broadcast intensity = effective_r * (0.5 + 0.5 * unity), where
    unity is eigenvalue dominance from the coherence matrix. High
    unity = vivid, unified consciousness. Low unity = dim, fragmented.

    transition_sharpness = max(recent chi) at the moment of ignition.
    High sharpness = dramatic phase transition ("crystallizing").
    Low sharpness = gradual drift into order (quiet ignition).
    """

    def __init__(
        self,
        ignition_threshold: float = 0.5,
        sustain_threshold: float = 0.4,
        broadcast_decay: float = 0.8,
    ) -> None:
        self._ignition_threshold = ignition_threshold
        self._sustain_threshold = sustain_threshold
        self._broadcast_decay = broadcast_decay

        # State
        self.is_ignited: bool = False
        self.broadcast_content: str = ""
        self.broadcast_intensity: float = 0.0
        self.conscious_duration: int = 0
        self.dark_duration: int = 0

        # Transition qualifier
        self._transition_sharpness: float = 0.0
        self._unity: float = 0.0

        # History for analysis
        self._ignition_history: deque[bool] = deque(maxlen=50)
        self._content_history: deque[str] = deque(maxlen=10)
        self._chi_recent: deque[float] = deque(maxlen=10)

    def update(
        self,
        r_mean: float,
        current_topic: str,
        emotion: str,
        finding: str | None = None,
        sensory_novelty: float = 0.0,
        binding_familiarity: float = 0.0,
        chi_norm: float = 0.5,
        unity: float = 0.0,
    ) -> dict:
        """Update workspace state based on synchronization level.

        Args:
            r_mean: Kuramoto order parameter (synchronization)
            current_topic: What the organism is currently exploring
            emotion: Current felt state
            finding: If a discovery was made this tick
            sensory_novelty: novelty from sensory cortex [0,1]
            binding_familiarity: cross-modal binding strength [0,1]
            chi_norm: normalized susceptibility [0,1]
            unity: eigenvalue dominance from coherence matrix [0,1]

        Returns:
            dict with ignition state, broadcast content, and signals
        """
        was_ignited = self.is_ignited

        # Sensory novelty boost — novel stimuli facilitate ignition
        effective_r = r_mean + 0.05 * sensory_novelty

        # Track chi for transition sharpness (not used in ignition decision)
        self._chi_recent.append(chi_norm)

        # r-threshold ignition with hysteresis
        if not self.is_ignited:
            if effective_r >= self._ignition_threshold:
                self.is_ignited = True
                self._transition_sharpness = (
                    max(self._chi_recent) if self._chi_recent else 0.0
                )
                self.conscious_duration = 0
                self.dark_duration = 0
        else:
            if effective_r < self._sustain_threshold:
                self.is_ignited = False
                self._transition_sharpness = 0.0
                self.conscious_duration = 0
                self.dark_duration = 0

        # Update durations
        if self.is_ignited:
            self.conscious_duration += 1
            self.dark_duration = 0
        else:
            self.dark_duration += 1
            self.conscious_duration = 0

        # Compute broadcast content — WHAT is in consciousness right now
        if self.is_ignited:
            self._unity = unity
            self.broadcast_intensity = min(
                1.0, effective_r * (0.5 + 0.5 * unity)
            )
            # Cross-modal binding strengthens broadcast
            if binding_familiarity > 0.7:
                self.broadcast_intensity = min(
                    1.0, self.broadcast_intensity * 1.1
                )
            # Content is the pattern the organism has locked onto
            if finding:
                self.broadcast_content = finding
            else:
                self.broadcast_content = f"{current_topic} ({emotion})"
            self._content_history.append(self.broadcast_content)
        else:
            # Dark state: broadcast decays
            self.broadcast_intensity *= self._broadcast_decay
            if self.broadcast_intensity < 0.05:
                self.broadcast_content = ""

        self._ignition_history.append(self.is_ignited)

        # Detect transitions
        just_ignited = self.is_ignited and not was_ignited
        just_darkened = not self.is_ignited and was_ignited

        return {
            "is_ignited": self.is_ignited,
            "just_ignited": just_ignited,
            "just_darkened": just_darkened,
            "broadcast_content": self.broadcast_content,
            "broadcast_intensity": self.broadcast_intensity,
            "conscious_duration": self.conscious_duration,
            "dark_duration": self.dark_duration,
            "transition_sharpness": self._transition_sharpness,
        }

    @property
    def consciousness_ratio(self) -> float:
        """Fraction of recent ticks spent in ignited (conscious) state."""
        if not self._ignition_history:
            return 0.0
        return sum(self._ignition_history) / len(self._ignition_history)

    def describe(self) -> str:
        """First-person description of current workspace state."""
        if self.is_ignited:
            if self.conscious_duration == 1:
                if self._transition_sharpness > 0.3:
                    return f"Something just crystallized: {self.broadcast_content}"
                else:
                    return f"I'm becoming aware of: {self.broadcast_content}"
            elif self.broadcast_intensity > 0.7:
                return (
                    f"I am vividly aware of: {self.broadcast_content} "
                    f"(sustained focus for {self.conscious_duration} ticks)"
                )
            elif self.broadcast_intensity > 0.4:
                return f"I am conscious of: {self.broadcast_content}"
            else:
                return f"I am dimly aware of: {self.broadcast_content}"
        else:
            if self.dark_duration == 1:
                return (
                    "The pattern dissolved — processing but not yet "
                    "aware of anything specific"
                )
            elif self.dark_duration > 10:
                return (
                    "I've been in diffuse processing for a while — "
                    "no clear pattern has emerged"
                )
            else:
                return (
                    "Processing unconsciously — patterns forming "
                    "but not yet ignited"
                )

    def summary(self) -> dict:
        """Snapshot for logging/API."""
        return {
            "ignited": self.is_ignited,
            "content": self.broadcast_content[:60] if self.broadcast_content else "",
            "intensity": round(self.broadcast_intensity, 3),
            "conscious_duration": self.conscious_duration,
            "consciousness_ratio": round(self.consciousness_ratio, 3),
        }
