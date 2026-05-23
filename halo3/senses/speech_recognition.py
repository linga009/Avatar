"""Speech Recognition — transcribes microphone audio to text using Whisper tiny.

Runs on CPU only (zero VRAM). Only called when sensory_stats.speech_detected is True
to avoid wasting compute on silence. Uses faster-whisper (CTranslate2 backend) for
efficient CPU inference.

The FNO gives Avatar the *experience* of hearing (spectral patterns).
Whisper gives Avatar *comprehension* (what was said).
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load whisper model on first use."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel("tiny", device="cpu", compute_type="int8")
            log.info("SpeechRecognizer: faster-whisper tiny loaded (CPU, int8)")
        except ImportError:
            log.warning("SpeechRecognizer: faster-whisper not installed — speech recognition disabled")
        except Exception as e:
            log.warning(f"SpeechRecognizer: failed to load model: {e}")
    return _model


def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe audio waveform to text.

    Args:
        audio: (N,) float32 waveform at sample_rate Hz
        sample_rate: audio sample rate (default 16kHz for Whisper)

    Returns:
        Transcribed text, or empty string if transcription fails or no speech detected.
    """
    model = _get_model()
    if model is None:
        return ""

    try:
        segments, info = model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text and len(text) > 2:
            log.debug(f"SpeechRecognizer: heard '{text[:80]}'")
            return text
    except Exception as e:
        log.debug(f"SpeechRecognizer: transcription failed: {e}")

    return ""
