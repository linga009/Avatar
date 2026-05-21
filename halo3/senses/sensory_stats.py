"""SensoryStatistics — tracks codebook activation dynamics for PFC interpretation.

The PFC reads sensory dynamics (flux, novelty, stability, cross-modal binding),
not raw codebook indices. This mirrors how biological PFC interfaces with
sensory cortex — it reads patterns, not individual neuron activations.
"""
from __future__ import annotations
import json
import logging
import os
from collections import defaultdict
import numpy as np

log = logging.getLogger(__name__)


class SensoryStatistics:
    """Tracks codebook activation patterns across ticks."""

    def __init__(self, audio_tokens: int, vision_tokens: int,
                 codebook_size: int, window: int = 20) -> None:
        self._audio_tokens = audio_tokens
        self._vision_tokens = vision_tokens
        self._codebook_size = codebook_size
        self._window = window
        self._prev_audio: np.ndarray | None = None
        self._prev_vision: np.ndarray | None = None
        self._cur_audio: np.ndarray | None = None
        self._cur_vision: np.ndarray | None = None
        self.audio_flux: int = 0
        self.vision_flux: int = 0
        self.audio_stability: int = 0
        self.vision_stability: int = 0
        self._audio_usage = np.zeros(codebook_size, dtype=np.float64)
        self._vision_usage = np.zeros(codebook_size, dtype=np.float64)
        self._total_ticks: int = 0
        self._cooccurrence: dict[tuple[int, int], int] = defaultdict(int)
        self._total_cooccurrences: int = 0
        self._audio_history: list[np.ndarray] = []
        self._vision_history: list[np.ndarray] = []

        # Speech detection (v3.8)
        self._speech_codes: set[int] = set()
        self.speech_detected: bool = False
        self.speech_stability: int = 0

    def register_speech_codes(self, codes: set[int]) -> None:
        """Register codebook indices that correspond to speech patterns."""
        self._speech_codes = codes

    def update(self, audio_indices, vision_indices) -> None:
        audio_np = np.array(audio_indices, dtype=np.int32)
        vision_np = np.array(vision_indices, dtype=np.int32)
        self._prev_audio = self._cur_audio
        self._prev_vision = self._cur_vision
        self._cur_audio = audio_np
        self._cur_vision = vision_np
        self._total_ticks += 1
        if self._prev_audio is not None:
            self.audio_flux = int(np.sum(audio_np != self._prev_audio))
            self.vision_flux = int(np.sum(vision_np != self._prev_vision))
        else:
            self.audio_flux = 0
            self.vision_flux = 0
        if self.audio_flux == 0 and self._prev_audio is not None:
            self.audio_stability += 1
        else:
            self.audio_stability = 0
        if self.vision_flux == 0 and self._prev_vision is not None:
            self.vision_stability += 1
        else:
            self.vision_stability = 0
        # Speech detection: >50% of active audio codes are in speech set
        if self._speech_codes and self._cur_audio is not None:
            speech_count = sum(1 for c in audio_np if int(c) in self._speech_codes)
            was_speech = self.speech_detected
            self.speech_detected = speech_count > len(audio_np) * 0.5
            if self.speech_detected and was_speech:
                self.speech_stability += 1
            elif not self.speech_detected:
                self.speech_stability = 0
        else:
            self.speech_detected = False
        for idx in audio_np:
            self._audio_usage[idx] += 1
        for idx in vision_np:
            self._vision_usage[idx] += 1
        self._audio_history.append(audio_np)
        self._vision_history.append(vision_np)
        if len(self._audio_history) > self._window:
            self._audio_history.pop(0)
            self._vision_history.pop(0)
        audio_dom = int(np.bincount(audio_np, minlength=self._codebook_size).argmax())
        vision_dom = int(np.bincount(vision_np, minlength=self._codebook_size).argmax())
        self._cooccurrence[(audio_dom, vision_dom)] += 1
        self._total_cooccurrences += 1

    @property
    def audio_novelty(self) -> float:
        if self._cur_audio is None or self._total_ticks == 0:
            return 0.0
        total = max(self._audio_usage.sum(), 1.0)
        freqs = self._audio_usage[self._cur_audio] / total
        return float(np.mean(1.0 - np.clip(freqs, 0, 1)))

    @property
    def vision_novelty(self) -> float:
        if self._cur_vision is None or self._total_ticks == 0:
            return 0.0
        total = max(self._vision_usage.sum(), 1.0)
        freqs = self._vision_usage[self._cur_vision] / total
        return float(np.mean(1.0 - np.clip(freqs, 0, 1)))

    @property
    def audio_dominant(self) -> int:
        if not self._audio_history:
            return 0
        return int(np.bincount(np.concatenate(self._audio_history), minlength=self._codebook_size).argmax())

    @property
    def vision_dominant(self) -> int:
        if not self._vision_history:
            return 0
        return int(np.bincount(np.concatenate(self._vision_history), minlength=self._codebook_size).argmax())

    @property
    def cross_modal_binding(self) -> float:
        if self._cur_audio is None or self._total_cooccurrences == 0:
            return 0.0
        audio_dom = int(np.bincount(self._cur_audio, minlength=self._codebook_size).argmax())
        vision_dom = int(np.bincount(self._cur_vision, minlength=self._codebook_size).argmax())
        pair_count = self._cooccurrence.get((audio_dom, vision_dom), 0)
        return min(1.0, pair_count / max(self._total_cooccurrences * 0.1, 1.0))

    @property
    def audio_usage_counts(self) -> np.ndarray:
        return self._audio_usage.copy()

    @property
    def vision_usage_counts(self) -> np.ndarray:
        return self._vision_usage.copy()

    def format_for_pfc(self) -> str:
        binding_label = "familiar" if self.cross_modal_binding > 0.5 else "novel"
        speech_str = ""
        if self._speech_codes:
            speech_str = f", speech={'yes' if self.speech_detected else 'no'}, speaking_for={self.speech_stability}"
        return (
            f"Senses: audio(flux={self.audio_flux}/{self._audio_tokens}, "
            f"novelty={self.audio_novelty:.2f}, stable={self.audio_stability}"
            f"{speech_str}), "
            f"vision(flux={self.vision_flux}/{self._vision_tokens}, "
            f"novelty={self.vision_novelty:.2f}, stable={self.vision_stability}), "
            f"binding={binding_label}({self.cross_modal_binding:.2f})"
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "audio_usage": self._audio_usage.tolist(),
            "vision_usage": self._vision_usage.tolist(),
            "total_ticks": self._total_ticks,
            "cooccurrence": {f"{k[0]},{k[1]}": v for k, v in self._cooccurrence.items()},
            "total_cooccurrences": self._total_cooccurrences,
            "audio_stability": self.audio_stability,
            "vision_stability": self.vision_stability,
            "speech_codes": list(self._speech_codes),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._audio_usage = np.array(data["audio_usage"])
            self._vision_usage = np.array(data["vision_usage"])
            self._total_ticks = data["total_ticks"]
            self._cooccurrence = defaultdict(int)
            for k, v in data.get("cooccurrence", {}).items():
                a, b = k.split(",")
                self._cooccurrence[(int(a), int(b))] = v
            self._total_cooccurrences = data.get("total_cooccurrences", 0)
            self.audio_stability = data.get("audio_stability", 0)
            self.vision_stability = data.get("vision_stability", 0)
            self._speech_codes = set(data.get("speech_codes", []))
            log.info(f"SensoryStatistics loaded from {path}")
        except Exception as e:
            log.warning(f"Failed to load sensory stats: {e}")
