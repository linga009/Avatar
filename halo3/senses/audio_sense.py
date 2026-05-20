"""AudioSense — Wav2Vec2-base encoder for always-on hearing.

Loads once at startup on CPU (~380 MB RAM).
Encodes a 2-second 16kHz audio chunk -> (8, 768) numpy float32.
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)

_N_FRAMES = 8         # temporal frames to extract
_SAMPLE_RATE = 16000  # Hz — Wav2Vec2 expects 16kHz
_MODEL_ID = "facebook/wav2vec2-base"

# Module-level imports so @patch can target them in tests
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
except ImportError:
    Wav2Vec2Processor = None  # type: ignore
    Wav2Vec2Model = None      # type: ignore


class AudioSense:
    """Wav2Vec2-base audio encoder. Call encode(audio_np) each tick."""

    def __init__(self, cache_dir: str = "data/model_cache") -> None:
        if Wav2Vec2Model is None:
            log.warning("transformers not installed. Hearing disabled.")
            self._available = False
            return
        try:
            log.info(f"Loading Wav2Vec2-base from {_MODEL_ID} (CPU)...")
            self._processor = Wav2Vec2Processor.from_pretrained(
                _MODEL_ID, cache_dir=cache_dir)
            self._model = Wav2Vec2Model.from_pretrained(
                _MODEL_ID, cache_dir=cache_dir)
            self._model.eval()
            log.info("AudioSense ready.")
            self._available = True
        except Exception as e:
            log.warning(f"AudioSense failed to load: {e}. Hearing disabled.")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def encode(self, audio_np: np.ndarray) -> np.ndarray | None:
        """Encode a mono float32 audio array at 16kHz.

        Args:
            audio_np: (N,) float32 array, values in [-1, 1]

        Returns:
            (8, 768) float32 numpy array, or None on failure.
        """
        if not self._available:
            return None
        try:
            return self._encode(audio_np)
        except Exception as e:
            log.warning(f"AudioSense encode error: {e}")
            return None

    def _encode(self, audio_np: np.ndarray) -> np.ndarray:
        """Internal encode — called by encode() and tests."""
        import torch
        inputs = self._processor(
            audio_np, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        # last_hidden_state: (1, T, 768) -> (T, 768)
        # np.asarray handles both torch tensors and numpy arrays (mocks)
        hidden = np.asarray(outputs.last_hidden_state[0])  # (T, 768)
        T = hidden.shape[0]
        # Sample N_FRAMES evenly spaced frames
        indices = np.linspace(0, T - 1, _N_FRAMES, dtype=int)
        return hidden[indices].astype(np.float32)  # (8, 768)
