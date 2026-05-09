"""Emotions — 2D valence derived from physics (surprise × confidence).

Maps the raw physics outputs (order parameter r, free energy delta)
into felt emotional states that modulate behavior.

Emotional space:
  High r + low surprise  → satisfaction
  High r + high surprise → pride (novel discovery confirmed)
  Low r  + high surprise → anxiety (overwhelmed)
  Low r  + low surprise  → boredom (nothing happening)
  Mid r  + any surprise  → curiosity (edge of understanding)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque


EMOTION_NAMES = ("satisfaction", "pride", "curiosity", "boredom", "anxiety")


@dataclass
class EmotionState:
    """Tracks current emotion and emotional history."""
    current: str = "curiosity"
    intensity: float = 0.5
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    _fe_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def update(self, r_mean: float, fe_delta: float) -> tuple[str, float]:
        """Compute emotion from physics. Returns (emotion_name, intensity)."""
        self._fe_history.append(abs(fe_delta))

        # Surprise: how unusual is this fe_delta relative to recent history
        if len(self._fe_history) > 1:
            mean_fe = sum(self._fe_history) / len(self._fe_history)
            surprise = min(1.0, abs(fe_delta) / (mean_fe + 1e-12))
        else:
            surprise = 0.5

        confidence = r_mean

        # Map to emotion
        if confidence > 0.6 and surprise < 0.4:
            emotion = "satisfaction"
            intensity = confidence * (1.0 - surprise)
        elif confidence > 0.6 and surprise >= 0.4:
            emotion = "pride"
            intensity = min(1.0, confidence * surprise)
        elif confidence < 0.35 and surprise > 0.5:
            emotion = "anxiety"
            intensity = min(1.0, surprise * (1.0 - confidence))
        elif confidence < 0.35 and surprise <= 0.5:
            emotion = "boredom"
            intensity = max(0.1, 1.0 - surprise - confidence)
        else:
            emotion = "curiosity"
            # Highest when r is near 0.5 (edge of synchronization)
            edge_factor = 1.0 - abs(r_mean - 0.5) * 2.0
            intensity = min(1.0, surprise * 0.4 + edge_factor * 0.6)

        self.current = emotion
        self.intensity = intensity
        self.history.append((emotion, intensity))
        return emotion, intensity

    @property
    def dominant_recent(self) -> str:
        """Most frequent emotion in last 20 ticks."""
        if len(self.history) < 2:
            return self.current
        recent = list(self.history)[-20:]
        counts = {}
        for e, _ in recent:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def emoji(self) -> str:
        """Single character representing current emotion."""
        return {
            "satisfaction": "😌",
            "pride": "✨",
            "curiosity": "🔍",
            "boredom": "😐",
            "anxiety": "⚡",
        }.get(self.current, "?")
