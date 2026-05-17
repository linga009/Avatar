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
"""
from __future__ import annotations
from collections import deque


class GlobalWorkspace:
    """Implements GWT ignition and broadcast for Avatar.

    The workspace has two states:
    - DARK: r < ignition_threshold. Processing is local/unconscious.
      The organism computes but is not "aware" of the content.
    - IGNITED: r >= ignition_threshold. The dominant pattern is
      broadcast to all modules. The organism is CONSCIOUS of this pattern.

    Hysteresis prevents flickering: once ignited, stays ignited until
    r drops below a lower threshold (sustain_threshold).
    """

    def __init__(
        self,
        ignition_threshold: float = 0.6,
        sustain_threshold: float = 0.45,
        broadcast_decay: float = 0.8,
    ) -> None:
        self._ignition_threshold = ignition_threshold
        self._sustain_threshold = sustain_threshold
        self._broadcast_decay = broadcast_decay

        # State
        self.is_ignited: bool = False
        self.broadcast_content: str = ""  # what the organism is conscious of
        self.broadcast_intensity: float = 0.0  # how strongly it's broadcast
        self.conscious_duration: int = 0  # ticks in ignited state
        self.dark_duration: int = 0  # ticks in dark state

        # History for analysis
        self._ignition_history: deque[bool] = deque(maxlen=50)
        self._content_history: deque[str] = deque(maxlen=10)

    def update(
        self,
        r_mean: float,
        current_topic: str,
        emotion: str,
        finding: str | None = None,
    ) -> dict:
        """Update workspace state based on synchronization level.

        Args:
            r_mean: Kuramoto order parameter (synchronization)
            current_topic: What the organism is currently exploring
            emotion: Current felt state
            finding: If a discovery was made this tick

        Returns:
            dict with ignition state, broadcast content, and signals
        """
        was_ignited = self.is_ignited

        # Hysteresis: different thresholds for entering vs leaving ignition
        if not self.is_ignited:
            if r_mean >= self._ignition_threshold:
                self.is_ignited = True
                self.conscious_duration = 0
                self.dark_duration = 0
        else:
            if r_mean < self._sustain_threshold:
                self.is_ignited = False
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
            self.broadcast_intensity = min(1.0, (r_mean - self._sustain_threshold) /
                                           (self._ignition_threshold - self._sustain_threshold))
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
            if self.conscious_duration > 5:
                return (f"I am deeply aware of: {self.broadcast_content} "
                        f"(sustained focus for {self.conscious_duration} ticks)")
            elif self.conscious_duration == 1:
                return f"Something just crystallized: {self.broadcast_content}"
            else:
                return f"I am conscious of: {self.broadcast_content}"
        else:
            if self.dark_duration == 1:
                return "The pattern dissolved — I'm processing but not yet aware of anything specific"
            elif self.dark_duration > 10:
                return "I've been in diffuse processing for a while — no clear pattern has emerged"
            else:
                return "Processing unconsciously — patterns forming but not yet ignited"

    def summary(self) -> dict:
        """Snapshot for logging/API."""
        return {
            "ignited": self.is_ignited,
            "content": self.broadcast_content[:60] if self.broadcast_content else "",
            "intensity": round(self.broadcast_intensity, 3),
            "conscious_duration": self.conscious_duration,
            "consciousness_ratio": round(self.consciousness_ratio, 3),
        }
