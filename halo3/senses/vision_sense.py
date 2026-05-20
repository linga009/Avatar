"""VisionSense — CLIP ViT-B/32 encoder for always-on vision.

Loads once at startup on CPU (~350 MB RAM).
Encodes a PIL Image -> (512,) numpy float32 CLS embedding.
"""
from __future__ import annotations
import logging
import numpy as np

log = logging.getLogger(__name__)

_MODEL_ID = "openai/clip-vit-base-patch32"

# Module-level imports so @patch can target them in tests
try:
    from transformers import CLIPProcessor, CLIPVisionModel
except ImportError:
    CLIPProcessor = None   # type: ignore
    CLIPVisionModel = None  # type: ignore


class VisionSense:
    """CLIP ViT-B/32 vision encoder. Call encode(pil_image) each tick."""

    def __init__(self, cache_dir: str = "data/model_cache") -> None:
        if CLIPVisionModel is None:
            log.warning("transformers not installed. Vision disabled.")
            self._available = False
            return
        try:
            log.info(f"Loading CLIP ViT-B/32 from {_MODEL_ID} (CPU)...")
            self._processor = CLIPProcessor.from_pretrained(
                _MODEL_ID, cache_dir=cache_dir)
            self._model = CLIPVisionModel.from_pretrained(
                _MODEL_ID, cache_dir=cache_dir)
            self._model.eval()
            log.info("VisionSense ready.")
            self._available = True
        except Exception as e:
            log.warning(f"VisionSense failed to load: {e}. Vision disabled.")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def encode(self, pil_image) -> np.ndarray | None:
        """Encode a PIL Image to a (512,) CLIP embedding.

        Returns:
            (512,) float32 numpy array, or None on failure.
        """
        if not self._available:
            return None
        try:
            return self._encode(pil_image)
        except Exception as e:
            log.warning(f"VisionSense encode error: {e}")
            return None

    def _encode(self, pil_image) -> np.ndarray:
        """Internal encode — called by encode() and tests."""
        import torch
        inputs = self._processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        # pooler_output: (1, 512) -> (512,)
        # np.asarray handles both torch tensors and numpy arrays (mocks)
        return np.asarray(outputs.pooler_output[0]).astype(np.float32)
