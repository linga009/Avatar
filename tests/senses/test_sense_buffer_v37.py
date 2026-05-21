"""Test SenseBuffer raw array loading for FNO pipeline."""
import json
import os
import time
import numpy as np
import pytest


def test_get_raw_arrays_returns_numpy(tmp_path):
    from halo3.senses.sense_buffer import SenseBuffer
    senses_dir = tmp_path / "senses"
    senses_dir.mkdir()
    audio = np.random.randn(32000).astype(np.float32)
    np.save(str(senses_dir / "audio_latest.npy"), audio)
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(str(senses_dir / "frame_latest.jpg"))
    with open(str(senses_dir / "meta.json"), "w") as f:
        json.dump({"has_audio": True, "has_video": True, "timestamp": time.time()}, f)
    buf = SenseBuffer(data_dir=str(tmp_path), stale_threshold_secs=30.0)
    result = buf.get_raw_arrays()
    assert result.audio_np is not None
    assert result.audio_np.shape == (32000,)
    assert result.vision_np is not None
    assert result.vision_np.shape == (224, 224, 3)
    assert result.vision_np.dtype == np.float32


def test_get_raw_arrays_stale_returns_none(tmp_path):
    from halo3.senses.sense_buffer import SenseBuffer
    senses_dir = tmp_path / "senses"
    senses_dir.mkdir()
    with open(str(senses_dir / "meta.json"), "w") as f:
        json.dump({"has_audio": True, "has_video": True, "timestamp": time.time() - 60}, f)
    buf = SenseBuffer(data_dir=str(tmp_path), stale_threshold_secs=30.0)
    result = buf.get_raw_arrays()
    assert result.audio_np is None
    assert result.vision_np is None


def test_get_raw_arrays_no_meta_returns_none(tmp_path):
    from halo3.senses.sense_buffer import SenseBuffer
    buf = SenseBuffer(data_dir=str(tmp_path), stale_threshold_secs=30.0)
    result = buf.get_raw_arrays()
    assert result.audio_np is None
    assert result.vision_np is None
