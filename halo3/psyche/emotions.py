"""Emotions — 2D valence derived from physics (surprise x confidence).

Maps the raw physics outputs (order parameter r, free energy delta)
into felt emotional states that modulate behavior.

Emotional space:
  High r + low surprise  -> satisfaction
  High r + high surprise -> pride (novel discovery confirmed)
  Low r  + high surprise -> anxiety (overwhelmed)
  Low r  + low surprise  -> boredom (nothing happening)
  Mid r  + any surprise  -> curiosity (edge of understanding)
  Repeated failure       -> frustration (dead-end, must change course)

v3.1 additions:
  - Frustration emotion: triggered by sustained zero results
  - Surprise recalibration: zero input when content expected = high surprise
  - Emotional inertia: exponential smoothing prevents tick-to-tick flipping
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque


EMOTION_NAMES = ("satisfaction", "pride", "curiosity", "boredom", "anxiety", "frustration")


@dataclass
class EmotionState:
    """Tracks current emotion and emotional history."""
    current: str = "curiosity"
    intensity: float = 0.5
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    _fe_history: deque = field(default_factory=lambda: deque(maxlen=50))
    # Emotional inertia: smoothed valence and arousal
    _valence: float = 0.0   # [-1, 1] negative=bad, positive=good
    _arousal: float = 0.5   # [0, 1] calm to excited

    def update(
        self,
        r_mean: float,
        fe_delta: float,
        perception_failed: bool = False,
        consecutive_failures: int = 0,
    ) -> tuple[str, float]:
        """Compute emotion from physics.

        Args:
            r_mean: Kuramoto order parameter (0-1)
            fe_delta: change in free energy
            perception_failed: True if last search returned zero results
            consecutive_failures: how many ticks in a row had zero results

        Returns (emotion_name, intensity).
        """
        self._fe_history.append(abs(fe_delta))

        # Surprise: how unusual is this fe_delta relative to recent history
        if len(self._fe_history) > 1:
            mean_fe = sum(self._fe_history) / len(self._fe_history)
            surprise = min(1.0, abs(fe_delta) / (mean_fe + 1e-12))
        else:
            surprise = 0.5

        # Recalibrate surprise for zero input: expecting content but getting
        # nothing IS surprising, not "low surprise"
        if perception_failed:
            surprise = max(surprise, 0.6)

        confidence = r_mean

        # --- Frustration override ---
        # Repeated failure is its own emotional state regardless of r or surprise.
        # This is the organism's immune response to dead-end attractors.
        if consecutive_failures >= 3:
            emotion = "frustration"
            intensity = min(1.0, 0.5 + consecutive_failures * 0.1)
        elif confidence > 0.6 and surprise < 0.4:
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
            edge_factor = 1.0 - abs(r_mean - 0.5) * 2.0
            intensity = min(1.0, surprise * 0.4 + edge_factor * 0.6)

        # --- Emotional inertia ---
        # Smooth transitions so emotions don't flip every tick.
        # Map current emotion to valence/arousal, then blend with previous.
        _emo_va = {
            "satisfaction": (0.7, 0.2),
            "pride":        (0.9, 0.8),
            "curiosity":    (0.3, 0.6),
            "boredom":      (-0.3, 0.1),
            "anxiety":      (-0.6, 0.9),
            "frustration":  (-0.8, 0.8),
        }
        new_v, new_a = _emo_va.get(emotion, (0.0, 0.5))
        alpha = 0.6  # how much new emotion influences (0=full inertia, 1=no inertia)
        self._valence = alpha * new_v + (1.0 - alpha) * self._valence
        self._arousal = alpha * new_a + (1.0 - alpha) * self._arousal

        # Re-derive emotion from smoothed valence/arousal (prevents flipping)
        # Only override if the smoothed state disagrees strongly
        smoothed = self._emotion_from_va(self._valence, self._arousal)
        if smoothed != emotion:
            # If raw and smoothed disagree, use smoothed unless raw is frustration
            # (frustration should punch through inertia)
            if emotion != "frustration":
                emotion = smoothed

        self.current = emotion
        self.intensity = intensity
        self.history.append((emotion, intensity))
        return emotion, intensity

    @staticmethod
    def _emotion_from_va(valence: float, arousal: float) -> str:
        """Map continuous valence/arousal back to discrete emotion."""
        if valence < -0.5 and arousal > 0.5:
            return "frustration" if valence < -0.7 else "anxiety"
        if valence < -0.1 and arousal < 0.3:
            return "boredom"
        if valence > 0.5 and arousal < 0.4:
            return "satisfaction"
        if valence > 0.5 and arousal >= 0.4:
            return "pride"
        return "curiosity"

    @property
    def dominant_recent(self) -> str:
        if len(self.history) < 2:
            return self.current
        recent = list(self.history)[-20:]
        counts = {}
        for e, _ in recent:
            counts[e] = counts.get(e, 0) + 1
        return max(counts, key=counts.get)

    def emoji(self) -> str:
        return {
            "satisfaction": "😌",
            "pride": "✨",
            "curiosity": "🔍",
            "boredom": "😐",
            "anxiety": "⚡",
            "frustration": "😤",
        }.get(self.current, "?")
