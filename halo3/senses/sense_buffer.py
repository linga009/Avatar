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

log = logging.getLogger(__name__)


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
