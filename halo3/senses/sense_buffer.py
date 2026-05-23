"""SenseBuffer — reads audio/frame files from the shared Docker volume.

The Windows host capture agent writes to data/senses/.
This module reads those files each tick and returns paths for the encoders.
"""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RawSenseData:
    """Raw numpy arrays for FNO processing, or None if unavailable."""
    audio_np: np.ndarray | None    # (32000,) float32
    vision_np: np.ndarray | None   # (224, 224, 3) float32


@dataclass
class RawSensePaths:
    """Paths to raw sense files, or None if unavailable/stale."""
    audio_path: str | None   # path to audio_latest.npy
    video_path: str | None   # path to frame_latest.jpg


class SenseBuffer:
    """Checks data/senses/ freshness and returns file paths for encoders."""

    def __init__(
        self,
        data_dir: str = "data",
        stale_threshold_secs: float = 30.0,
    ) -> None:
        self._senses_dir = os.path.join(data_dir, "senses")
        self._stale_threshold = stale_threshold_secs

    def get_raw(self) -> RawSensePaths:
        """Return file paths if fresh, None fields if stale or missing."""
        meta_path = os.path.join(self._senses_dir, "meta.json")
        if not os.path.exists(meta_path):
            return RawSensePaths(None, None)

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            log.warning(f"SenseBuffer: failed to read meta.json: {e}")
            return RawSensePaths(None, None)

        age = time.time() - meta.get("timestamp", 0)
        if age > self._stale_threshold:
            return RawSensePaths(None, None)

        audio_path = None
        if meta.get("has_audio"):
            p = os.path.join(self._senses_dir, "audio_latest.npy")
            if os.path.exists(p):
                audio_path = p

        video_path = None
        if meta.get("has_video"):
            p = os.path.join(self._senses_dir, "frame_latest.jpg")
            if os.path.exists(p):
                video_path = p

        return RawSensePaths(audio_path, video_path)

    def get_raw_arrays(self) -> RawSenseData:
        """Return raw numpy arrays for FNO processing."""
        paths = self.get_raw()
        audio_np = None
        vision_np = None
        if paths.audio_path is not None:
            try:
                audio_np = np.load(paths.audio_path).astype(np.float32)
            except Exception as e:
                log.warning(f"SenseBuffer: failed to load audio: {e}")
        if paths.video_path is not None:
            try:
                from PIL import Image
                img = Image.open(paths.video_path).convert("RGB")
                vision_np = np.array(img, dtype=np.float32) / 255.0
            except Exception as e:
                log.warning(f"SenseBuffer: failed to load image: {e}")
        return RawSenseData(audio_np, vision_np)

    def archive_audio(self, audio_np: np.ndarray, tick: int,
                      max_archive: int = 50) -> None:
        """Save audio snapshot to rolling archive for dream visitor replay."""
        archive_dir = os.path.join(self._senses_dir, "audio_archive")
        os.makedirs(archive_dir, exist_ok=True)
        np.save(os.path.join(archive_dir, f"tick_{tick:06d}.npy"), audio_np)
        # Prune oldest if over limit
        files = sorted(f for f in os.listdir(archive_dir) if f.endswith(".npy"))
        while len(files) > max_archive:
            os.remove(os.path.join(archive_dir, files.pop(0)))

    @staticmethod
    def load_audio_archive(data_dir: str = "data") -> list[tuple[str, np.ndarray]]:
        """Load all archived audio snapshots. Returns list of (filename, audio_array)."""
        archive_dir = os.path.join(data_dir, "senses", "audio_archive")
        if not os.path.exists(archive_dir):
            return []
        pairs = []
        for f in sorted(os.listdir(archive_dir)):
            if f.endswith(".npy"):
                try:
                    audio = np.load(os.path.join(archive_dir, f)).astype(np.float32)
                    pairs.append((f, audio))
                except Exception:
                    pass
        return pairs
