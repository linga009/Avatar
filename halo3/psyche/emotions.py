"""Emotions — affect derived from COP phase-diagram geometry.

Maps (r, chi_norm, f_dot) to felt emotional states.

COP replaces the if/elif threshold tree with manifold position:
  High r + low chi + resolving -> satisfaction (ordered, calm)
  High r + high chi + resolving -> pride (ordered + sensitive)
  Mid r + high chi -> curiosity (at the critical edge)
  Low r + low chi -> boredom (disordered, rigid)
  Low r + high chi + worsening -> anxiety (disordered, reactive)
  Sustained failure -> frustration (punches through)

Emotional inertia (valence/arousal EMA) preserved from v3.11.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque


EMOTION_NAMES = ("satisfaction", "pride", "curiosity", "boredom", "anxiety",
                 "frustration", "flow", "exhaustion")


@dataclass
class EmotionState:
    """Tracks current emotion and emotional history."""
    current: str = "curiosity"
    intensity: float = 0.5
    qualifier: str = ""
    mood: str = "settling"
    body_event: str = ""
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    _valence: float = 0.0
    _arousal: float = 0.5

    def update(
        self,
        r_mean: float,
        fe_delta: float,
        chi_norm: float = 0.5,
        perception_failed: bool = False,
        consecutive_failures: int = 0,
        sensory_novelty: float = 0.0,
        sensory_stability: int = 0,
        speech_detected: bool = False,
        dF_dt: float = 0.0,
        f_thermo_flat_ticks: int = 0,
        tau_norm: float = 0.5,
        unity: float = 0.5,
        is_ignited: bool = False,
        just_ignited: bool = False,
        avalanche_just_ended: bool = False,
        self_surprise: float = 0.0,
    ) -> tuple[str, str, float]:
        """Compute emotion from COP phase-diagram position.

        Args:
            r_mean: Kuramoto order parameter (0-1)
            fe_delta: free energy change this tick
            chi_norm: normalized susceptibility from COP engine (0-1)
            perception_failed: True if search returned zero results
            consecutive_failures: ticks in a row with zero results
            sensory_novelty: from sensory stats (0-1)
            sensory_stability: consecutive stable ticks
            speech_detected: whether speech was heard
            dF_dt: rate of change of free energy
            f_thermo_flat_ticks: ticks F_thermo has been flat
            tau_norm: normalized relaxation time from COP engine (0-1)
            unity: coherence unity index from COP engine (0-1)
            is_ignited: True when SOC is in ignited (critical) state
            just_ignited: True on the tick ignition first occurred
            avalanche_just_ended: True on the tick an avalanche ended
            self_surprise: prediction error magnitude (0-1)

        Returns (emotion_name, qualifier, intensity).
        """
        # Clear transient body_event at the start of each tick
        self.body_event = ""
        f_dot = -fe_delta  # positive when surprise is resolving

        # Sensory novelty amplifies openness
        effective_chi = min(1.0, chi_norm + 0.15 * sensory_novelty
                           if sensory_novelty > 0.8 else chi_norm)

        # --- Metabolic flow: highest priority when conditions met ---
        # chi > 0.3, F decreasing rapidly = efficient learning
        if effective_chi > 0.3 and dF_dt < -100.0:
            emotion = "flow"
            intensity = min(1.0, effective_chi * 0.5 + min(1.0, abs(dF_dt) / 5000.0) * 0.5)
        # --- Thermodynamic exhaustion: F flat for 20+ ticks, still responsive ---
        elif f_thermo_flat_ticks >= 20 and effective_chi > 0.2:
            emotion = "exhaustion"
            intensity = min(1.0, 0.4 + f_thermo_flat_ticks * 0.02)
        # --- Frustration split: growth vs futile ---
        elif consecutive_failures >= 3 or (r_mean < 0.2 and consecutive_failures >= 2):
            if dF_dt < 0:
                # F decreasing = productive struggle (growth frustration)
                emotion = "frustration"
                intensity = min(1.0, 0.4 + consecutive_failures * 0.08)
            else:
                # F flat or increasing = futile struggle
                emotion = "frustration"
                intensity = min(1.0, 0.6 + consecutive_failures * 0.1)
        # --- COP manifold regions ---
        elif r_mean > 0.55 and effective_chi < 0.4 and f_dot > 0.005:
            emotion = "satisfaction"
            intensity = min(1.0, r_mean * (1.0 - effective_chi))
        elif r_mean > 0.55 and effective_chi >= 0.4 and f_dot > 0.005:
            emotion = "pride"
            intensity = min(1.0, r_mean * effective_chi)
        elif r_mean < 0.35 and effective_chi > 0.5 and f_dot < -0.005:
            emotion = "anxiety"
            intensity = min(1.0, effective_chi * (1.0 - r_mean))
        elif r_mean < 0.35 and effective_chi < 0.3:
            emotion = "boredom"
            intensity = max(0.1, 1.0 - effective_chi - r_mean)
        else:
            emotion = "curiosity"
            edge_factor = 1.0 - abs(r_mean - 0.5) * 2.0
            intensity = min(1.0, effective_chi * 0.6 + edge_factor * 0.4)

        # --- Emotional inertia (EMA smoothing) ---
        _emo_va = {
            "satisfaction": (0.7, 0.2),
            "pride":        (0.9, 0.8),
            "curiosity":    (0.3, 0.6),
            "boredom":      (-0.3, 0.1),
            "anxiety":      (-0.6, 0.9),
            "frustration":  (-0.8, 0.8),
            "flow":         (0.8, 0.7),   # positive, high arousal — in the zone
            "exhaustion":   (-0.1, 0.15),  # neutral valence, very low arousal
        }
        new_v, new_a = _emo_va.get(emotion, (0.0, 0.5))
        alpha = 0.6
        self._valence = alpha * new_v + (1.0 - alpha) * self._valence
        self._arousal = alpha * new_a + (1.0 - alpha) * self._arousal

        if sensory_stability > 3:
            self._arousal *= 0.9
        if speech_detected:
            self._valence = min(1.0, self._valence + 0.05)

        smoothed = self._emotion_from_va(self._valence, self._arousal)
        if smoothed != emotion and emotion != "frustration":
            emotion = smoothed

        # --- COP-derived qualifier ---
        qualifier = self._compute_qualifier(emotion, chi_norm, tau_norm, unity, dF_dt)

        # --- Mood from ignition state ---
        if just_ignited:
            self.mood = "awakening"
        elif is_ignited:
            self.mood = "clarity"
        elif chi_norm > 0.4:
            self.mood = "threshold"
        else:
            self.mood = "settling"

        # --- Transient body events ---
        if avalanche_just_ended:
            self.body_event = "release"
        elif just_ignited:
            self.body_event = "surfacing"
        elif self_surprise > 0.5:
            self.body_event = "jolt"

        self.current = emotion
        self.qualifier = qualifier
        self.intensity = intensity
        self.history.append((emotion, intensity))
        return emotion, qualifier, intensity

    @staticmethod
    def _compute_qualifier(
        emotion: str,
        chi: float,
        tau: float,
        unity: float,
        dF_dt: float,
    ) -> str:
        """Map (emotion, COP state) to a sub-type adjective qualifier."""
        if emotion == "curiosity":
            if chi > 0.6 and dF_dt < -50:
                return "burning"
            if chi > 0.6 and abs(dF_dt) < 50:
                return "watchful"
            if chi < 0.3:
                return "restless"
            return "open"
        if emotion == "satisfaction":
            if unity > 0.7:
                return "deep"
            if unity < 0.4:
                return "partial"
            return "warm"
        if emotion == "pride":
            if chi > 0.6:
                return "luminous"
            return "quiet"
        if emotion == "frustration":
            if dF_dt < 0:
                return "growing"
            return "futile"
        if emotion == "anxiety":
            if tau > 0.7:
                return "creeping"
            if tau < 0.3:
                return "sharp"
            return "tight"
        if emotion == "boredom":
            if chi < 0.15:
                return "numb"
            return "dull"
        if emotion == "flow":
            return "effortless"
        if emotion == "exhaustion":
            return "heavy"
        return ""

    @staticmethod
    def _emotion_from_va(valence: float, arousal: float) -> str:
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
            "satisfaction": "\U0001f60c",
            "pride": "\u2728",
            "curiosity": "\U0001f50d",
            "boredom": "\U0001f610",
            "anxiety": "\u26a1",
            "frustration": "\U0001f624",
            "flow": "\U0001f525",       # fire — in the zone
            "exhaustion": "\U0001f6b6",  # walking — depleted, seeking new ground
        }.get(self.current, "?")
