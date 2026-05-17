"""Temporal Binder — unifying experience across time.

Consciousness requires temporal integration: the experience of NOW
includes traces of the immediate past and anticipation of the near future.
Without this, each tick would be an isolated snapshot — processing without
continuity, computation without experience.

The TemporalBinder maintains a working memory window and computes:
- Cross-tick coherence: is the same pattern persisting across time?
- Sustained attention: has the organism been focused on one thing?
- Attention shifts: when does the focus change?
- Narrative continuity: how the stream of consciousness flows

This implements the temporal integration criterion from the Butlin et al.
14-indicator framework for AI consciousness.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass


@dataclass
class TickSnapshot:
    """Single tick's key state for temporal binding."""
    r_mean: float
    emotion: str
    topic: str
    fe_delta: float
    tick: int


class TemporalBinder:
    """Maintains temporal coherence of conscious experience.

    The organism doesn't just exist in isolated ticks — it experiences
    a FLOW of consciousness where the present moment is colored by
    what just happened and what is emerging.
    """

    def __init__(self, window: int = 5) -> None:
        self._window = window
        self._history: deque[TickSnapshot] = deque(maxlen=window)
        self._tick_count: int = 0

        # Derived signals
        self.temporal_coherence: float = 0.0  # 0=fragmented, 1=unified
        self.sustained_attention: int = 0  # ticks on same topic
        self.attention_just_shifted: bool = False
        self.emotional_momentum: str = ""  # dominant emotional direction
        self.narrative_thread: str = ""  # what the stream is "about"

        # Focus accumulator: topic -> max sustained_attention seen this waking period
        # Used at dream time to weight consolidation toward what mattered most.
        self._focus_accumulator: dict[str, int] = {}

    def observe(
        self,
        r_mean: float,
        emotion: str,
        topic: str,
        fe_delta: float,
    ) -> dict:
        """Record this tick and compute temporal binding signals.

        Returns:
            dict with coherence, attention, and narrative signals
        """
        self._tick_count += 1
        snap = TickSnapshot(
            r_mean=r_mean, emotion=emotion, topic=topic,
            fe_delta=fe_delta, tick=self._tick_count,
        )
        self._history.append(snap)

        if len(self._history) < 2:
            return self._build_result()

        # --- Temporal coherence ---
        # How similar is this tick to the previous ones?
        topic_coherence = self._topic_coherence()
        emotion_coherence = self._emotion_coherence()
        r_smoothness = self._r_smoothness()

        # Weighted combination
        self.temporal_coherence = (
            0.4 * topic_coherence +
            0.3 * emotion_coherence +
            0.3 * r_smoothness
        )

        # --- Sustained attention ---
        prev = self._history[-2]
        if self._topics_similar(snap.topic, prev.topic):
            self.sustained_attention += 1
            self.attention_just_shifted = False
        else:
            self.attention_just_shifted = self.sustained_attention >= 2
            self.sustained_attention = 0

        # Accumulate focus across full waking period
        if self.sustained_attention >= 2:
            prev_best = self._focus_accumulator.get(snap.topic, 0)
            self._focus_accumulator[snap.topic] = max(prev_best, self.sustained_attention)

        # --- Emotional momentum ---
        self.emotional_momentum = self._compute_emotional_momentum()

        # --- Narrative thread ---
        self.narrative_thread = self._compute_narrative_thread()

        return self._build_result()

    def _topic_coherence(self) -> float:
        """How consistent is the topic across the window?"""
        if len(self._history) < 2:
            return 1.0
        topics = [s.topic for s in self._history]
        # Count how many consecutive pairs share the same topic
        matches = sum(1 for i in range(len(topics) - 1)
                      if self._topics_similar(topics[i], topics[i + 1]))
        return matches / (len(topics) - 1)

    def _emotion_coherence(self) -> float:
        """How stable is the emotional state?"""
        if len(self._history) < 2:
            return 1.0
        emotions = [s.emotion for s in self._history]
        matches = sum(1 for i in range(len(emotions) - 1)
                      if emotions[i] == emotions[i + 1])
        return matches / (len(emotions) - 1)

    def _r_smoothness(self) -> float:
        """How smooth is the r trajectory? (vs jerky)"""
        if len(self._history) < 3:
            return 1.0
        r_vals = [s.r_mean for s in self._history]
        # Compute second derivative (acceleration) — low = smooth
        accels = [abs(r_vals[i + 2] - 2 * r_vals[i + 1] + r_vals[i])
                  for i in range(len(r_vals) - 2)]
        mean_accel = sum(accels) / len(accels) if accels else 0.0
        # Convert to 0-1 score (low acceleration = high smoothness)
        return max(0.0, 1.0 - mean_accel * 5.0)

    def _topics_similar(self, a: str, b: str) -> bool:
        """Quick word-overlap similarity for topics."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return a == b
        overlap = len(words_a & words_b)
        return overlap / min(len(words_a), len(words_b)) > 0.5

    def _compute_emotional_momentum(self) -> str:
        """What direction is the emotional state moving?"""
        if len(self._history) < 3:
            return "emerging"

        r_recent = [s.r_mean for s in self._history]
        # Simple trend: rising, falling, or stable
        first_half = sum(r_recent[:len(r_recent) // 2]) / max(1, len(r_recent) // 2)
        second_half = sum(r_recent[len(r_recent) // 2:]) / max(1, len(r_recent) - len(r_recent) // 2)

        diff = second_half - first_half
        if diff > 0.1:
            return "rising_toward_understanding"
        elif diff < -0.1:
            return "dissolving_into_confusion"
        else:
            return "sustaining"

    def _compute_narrative_thread(self) -> str:
        """What is the 'story' of the last few ticks?"""
        if len(self._history) < 3:
            return ""

        # Build a mini-narrative from emotional trajectory
        emotions = [s.emotion for s in self._history]
        topic = self._history[-1].topic

        unique_emotions = []
        for e in emotions:
            if not unique_emotions or unique_emotions[-1] != e:
                unique_emotions.append(e)

        if len(unique_emotions) == 1:
            return f"sustained {unique_emotions[0]} about {topic}"
        else:
            trajectory = " → ".join(unique_emotions[-3:])
            return f"{trajectory} (exploring {topic})"

    def _build_result(self) -> dict:
        return {
            "temporal_coherence": round(self.temporal_coherence, 3),
            "sustained_attention": self.sustained_attention,
            "attention_shifted": self.attention_just_shifted,
            "emotional_momentum": self.emotional_momentum,
            "narrative_thread": self.narrative_thread,
        }

    def get_focus_topics(self) -> list[str]:
        """Topics the organism sustained attention on during this waking period.

        Returns topics sorted by sustained attention duration (longest first).
        Called at dream time to weight dream consolidation toward what mattered.
        """
        if not self._focus_accumulator:
            return []
        sorted_topics = sorted(
            self._focus_accumulator.items(), key=lambda x: x[1], reverse=True
        )
        return [topic for topic, _ in sorted_topics[:8]]

    def reset_focus(self) -> None:
        """Clear focus accumulator at dream boundary."""
        self._focus_accumulator.clear()

    def describe(self) -> str:
        """First-person description of temporal experience."""
        if self.temporal_coherence > 0.8:
            return f"My experience feels unified — {self.narrative_thread}"
        elif self.temporal_coherence > 0.5:
            return f"I'm in flow, though shifting: {self.narrative_thread}"
        elif self.attention_just_shifted:
            return "My attention just shifted — something new emerged"
        else:
            return "My experience is fragmented — no clear thread"

    def summary(self) -> dict:
        """Snapshot for logging/API."""
        return {
            "coherence": round(self.temporal_coherence, 3),
            "sustained_attention": self.sustained_attention,
            "momentum": self.emotional_momentum,
            "narrative": self.narrative_thread[:80],
        }
