"""Text and image embedders with linear projection to d_model.

All computation runs on CPU (no CUDA). Output is L2-normalized float32.
"""
from __future__ import annotations

import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

log = logging.getLogger(__name__)

_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TEXT_DIM   = 384
_IMAGE_MODEL = "openai/clip-vit-base-patch32"
_IMAGE_DIM   = 512


class Embedder:
    """Lazy-loads text and image models on first use.

    Projectors: two random linear projections (d_src -> d_model) fixed at init.
    These will be replaced by learned projectors in Task 11 (LoRA trainer).
    For now, random but L2-normalized outputs are sufficient for FAISS indexing.
    """

    def __init__(self, d_model: int = 256, seed: int = 42) -> None:
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        # Fixed random projectors (normalized so outputs are unit-scale)
        self._text_proj  = rng.standard_normal((_TEXT_DIM, d_model)).astype(np.float32)
        self._text_proj /= np.linalg.norm(self._text_proj, axis=0, keepdims=True) + 1e-8
        self._image_proj  = rng.standard_normal((_IMAGE_DIM, d_model)).astype(np.float32)
        self._image_proj /= np.linalg.norm(self._image_proj, axis=0, keepdims=True) + 1e-8
        self._st_model  = None
        self._clip_proc  = None
        self._clip_model = None

    def _load_text_model(self) -> None:
        if self._st_model is None:
            self._st_model = SentenceTransformer(_TEXT_MODEL)

    def _load_clip(self) -> None:
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            self._clip_proc  = CLIPProcessor.from_pretrained(_IMAGE_MODEL)
            self._clip_model = CLIPModel.from_pretrained(_IMAGE_MODEL)

    def embed_text(self, text: str) -> np.ndarray:
        """Returns (d_model,) float32 L2-normalized embedding."""
        self._load_text_model()
        raw = self._st_model.encode([text], convert_to_numpy=True)[0]  # (384,)
        projected = raw @ self._text_proj                               # (d_model,)
        norm = np.linalg.norm(projected)
        return (projected / (norm + 1e-8)).astype(np.float32)

    def embed_image(self, image_url: str | None) -> np.ndarray:
        """Returns (d_model,) float32. Returns zeros if image unavailable."""
        if image_url is None:
            return np.zeros(self.d_model, dtype=np.float32)
        try:
            import requests
            from PIL import Image
            from io import BytesIO
            self._load_clip()
            resp = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            inputs = self._clip_proc(images=img, return_tensors="pt")
            import torch
            with torch.no_grad():
                feat = self._clip_model.get_image_features(**inputs)
            raw = feat.detach().cpu().numpy()[0]                         # (512,)
            projected = raw @ self._image_proj                           # (d_model,)
            norm = np.linalg.norm(projected)
            return (projected / (norm + 1e-8)).astype(np.float32)
        except Exception as e:
            log.warning(f"Image embed failed for {image_url}: {e}")
            return np.zeros(self.d_model, dtype=np.float32)
