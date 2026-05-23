"""TTS Self-Narration — Avatar reads its own text aloud for speech-text pairing.

Phase B: espeak-ng (rule-based, <5MB, clean phonemes)
Phase C: piper (neural, ~50MB, natural prosody) — v3.10
"""
from __future__ import annotations
import logging
import subprocess
import tempfile
import os
import numpy as np

log = logging.getLogger(__name__)


def extract_narration_text(texts: list[str], max_words: int = 20) -> str:
    """Extract first ~max_words from perception texts for TTS narration."""
    if not texts:
        return ""
    combined = " ".join(texts)
    words = []
    for w in combined.split():
        cleaned = "".join(c for c in w if c.isalnum() or c in "'-")
        if cleaned and len(cleaned) > 1:
            words.append(cleaned)
        if len(words) >= max_words:
            break
    return " ".join(words)


class TTSNarrator:
    """Text-to-speech narration for paired speech-text training."""

    def __init__(self, mode: str = "piper", sample_rate: int = 16000,
                 duration_samples: int = 32000) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._duration = duration_samples
        self._available = self._check_available()

    def _check_available(self) -> bool:
        if self._mode == "piper":
            try:
                result = subprocess.run(
                    ["piper", "--version"],
                    capture_output=True, timeout=5)
                if result.returncode == 0:
                    log.info("TTSNarrator: piper available (neural TTS)")
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                log.info("TTSNarrator: piper not found, falling back to espeak-ng")
            # Fallback to espeak
            self._mode = "espeak"

        if self._mode == "espeak":
            try:
                result = subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True, timeout=5)
                ok = result.returncode == 0
                if ok:
                    log.info("TTSNarrator: espeak-ng available")
                return ok
            except (FileNotFoundError, subprocess.TimeoutExpired):
                log.warning("TTSNarrator: espeak-ng not found. TTS disabled.")
                return False
        return False

    @property
    def available(self) -> bool:
        return self._available

    def narrate(self, text: str) -> np.ndarray:
        """Convert text to audio waveform. Returns (duration_samples,) float32."""
        if not text or not self._available:
            return np.zeros(self._duration, dtype=np.float32)
        try:
            if self._mode == "piper":
                return self._narrate_piper(text)
            elif self._mode == "espeak":
                return self._narrate_espeak(text)
        except Exception as e:
            log.warning(f"TTS narration failed: {e}")
        return np.zeros(self._duration, dtype=np.float32)

    def _narrate_piper(self, text: str) -> np.ndarray:
        """Generate audio using Piper neural TTS."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            # Piper reads text from stdin, writes WAV to --output_file
            proc = subprocess.run(
                ["piper", "--model", "/app/data/piper/en_US-lessac-medium.onnx",
                 "--output_file", tmp_path],
                input=text.encode("utf-8"),
                capture_output=True, timeout=10)
            if proc.returncode != 0:
                log.warning(f"Piper failed (rc={proc.returncode}), falling back to espeak")
                return self._narrate_espeak(text)
            return self._load_wav(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _narrate_espeak(self, text: str) -> np.ndarray:
        """Generate audio using espeak-ng."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            subprocess.run(
                ["espeak-ng", "-w", tmp_path, "-s", "150", text],
                capture_output=True, timeout=10, check=True)
            return self._load_wav(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _load_wav(self, path: str) -> np.ndarray:
        """Load a WAV file and resample/pad to target duration."""
        import wave
        with wave.open(path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            sample_width = wf.getsampwidth()
            orig_rate = wf.getframerate()
        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 1:
            audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            return np.zeros(self._duration, dtype=np.float32)
        if orig_rate != self._sample_rate:
            ratio = self._sample_rate / orig_rate
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
            audio = audio[indices]
        if len(audio) >= self._duration:
            audio = audio[:self._duration]
        else:
            audio = np.pad(audio, (0, self._duration - len(audio)))
        return audio.astype(np.float32)
