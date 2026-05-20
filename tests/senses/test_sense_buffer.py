"""SenseBuffer unit tests — no real files needed, uses tmp_path."""
import json
import time
import numpy as np
import pytest
from PIL import Image


def write_sense_files(tmp_path, with_audio=True, with_video=True, age_secs=0):
    senses_dir = tmp_path / "senses"
    senses_dir.mkdir()

    timestamp = time.time() - age_secs
    meta = {"has_audio": with_audio, "has_video": with_video, "timestamp": timestamp}
    (senses_dir / "meta.json").write_text(json.dumps(meta))

    if with_audio:
        audio = np.zeros(32000, dtype=np.float32)
        np.save(str(senses_dir / "audio_latest.npy"), audio)

    if with_video:
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img.save(str(senses_dir / "frame_latest.jpg"))

    return senses_dir


def test_sense_buffer_returns_none_when_stale(tmp_path):
    write_sense_files(tmp_path, age_secs=60)  # 60s old — stale
    from halo3.senses.sense_buffer import SenseBuffer
    buf = SenseBuffer(data_dir=str(tmp_path), stale_threshold_secs=30)
    feats = buf.get_raw()
    assert feats.audio_path is None
    assert feats.video_path is None


def test_sense_buffer_returns_paths_when_fresh(tmp_path):
    write_sense_files(tmp_path, age_secs=5)  # 5s old — fresh
    from halo3.senses.sense_buffer import SenseBuffer
    buf = SenseBuffer(data_dir=str(tmp_path), stale_threshold_secs=30)
    feats = buf.get_raw()
    assert feats.audio_path is not None
    assert feats.video_path is not None


def test_sense_buffer_no_meta(tmp_path):
    (tmp_path / "senses").mkdir()
    from halo3.senses.sense_buffer import SenseBuffer
    buf = SenseBuffer(data_dir=str(tmp_path), stale_threshold_secs=30)
    feats = buf.get_raw()
    assert feats.audio_path is None
    assert feats.video_path is None
