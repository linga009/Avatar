"""Windows host capture agent for Avatar senses.

Records microphone + camera and writes to the Docker shared volume.
Run from anywhere: python capture_agent/capture_agent.py

The Docker volume is .worktrees/halo3/data/senses/ which maps to /app/data/senses/.
"""
from __future__ import annotations
import json
import os
import time
import threading
import numpy as np

# Write directly to the worktree path that Docker mounts
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
SENSES_DIR = os.path.join(_PROJECT_ROOT, ".worktrees", "halo3", "data", "senses")

SAMPLE_RATE = 16000       # Hz — Wav2Vec2 expects 16kHz
AUDIO_CHUNK_SECS = 2      # seconds of audio per chunk
FRAME_INTERVAL_SECS = 10  # seconds between camera captures
MOTION_THRESHOLD = 30     # pixel diff threshold for motion capture


import shutil
import tempfile

# Temp dir for staging files before copying into the Docker-mounted senses dir
_STAGE_DIR = os.path.join(tempfile.gettempdir(), "avatar_senses_stage")
os.makedirs(_STAGE_DIR, exist_ok=True)


def _safe_write(dst: str, data: bytes) -> None:
    """Write data to dst via a temp file, retrying on Windows/Docker locks."""
    # Stage in temp dir (never locked by Docker)
    stage = os.path.join(_STAGE_DIR, os.path.basename(dst) + ".tmp")
    with open(stage, "wb") as f:
        f.write(data)
    # Copy into Docker-mounted dir with retries
    for _ in range(10):
        try:
            shutil.copy2(stage, dst)
            return
        except PermissionError:
            time.sleep(0.2)
    # Last attempt
    try:
        shutil.copy2(stage, dst)
    except PermissionError:
        pass  # Skip this cycle rather than crash


def _write_meta(has_audio: bool, has_video: bool) -> None:
    meta = {"has_audio": has_audio, "has_video": has_video, "timestamp": time.time()}
    _safe_write(
        os.path.join(SENSES_DIR, "meta.json"),
        json.dumps(meta).encode()
    )


def audio_loop(stop_event: threading.Event) -> None:
    """Continuously record 2s audio chunks at 16kHz, save to shared volume."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed — audio disabled. Run: pip install sounddevice")
        return

    print(f"Audio: recording at {SAMPLE_RATE} Hz, {AUDIO_CHUNK_SECS}s chunks")
    audio_path = os.path.join(SENSES_DIR, "audio_latest.npy")

    while not stop_event.is_set():
        try:
            chunk = sd.rec(
                int(AUDIO_CHUNK_SECS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio_mono = chunk[:, 0]  # (32000,)
            # Save to stage dir then copy (avoids Docker file locks)
            stage = os.path.join(_STAGE_DIR, "audio_latest.npy")
            np.save(stage, audio_mono)
            _safe_write(audio_path, open(stage, "rb").read())
        except Exception as e:
            print(f"Audio capture error: {e}")
            time.sleep(2)


def vision_loop(stop_event: threading.Event) -> None:
    """Capture camera frames on schedule or motion, save to shared volume."""
    try:
        import cv2
    except ImportError:
        print("opencv-python not installed — vision disabled. Run: pip install opencv-python")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Vision: no camera found — vision disabled")
        return

    print(f"Vision: capturing every {FRAME_INTERVAL_SECS}s (or on motion)")
    frame_path = os.path.join(SENSES_DIR, "frame_latest.jpg")
    prev_gray = None
    last_capture = 0.0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now = time.time()
        motion = False

        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float)))
            motion = diff > MOTION_THRESHOLD

        if motion or (now - last_capture) >= FRAME_INTERVAL_SECS:
            # Resize to 224x224 (CLIP input size)
            resized = cv2.resize(frame, (224, 224))
            # Convert BGR -> RGB then back for imwrite
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            stage = os.path.join(_STAGE_DIR, "frame_latest.jpg")
            cv2.imwrite(stage, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            _safe_write(frame_path, open(stage, "rb").read())
            last_capture = now
            if motion:
                print(f"Vision: motion detected, captured frame")

        prev_gray = gray
        time.sleep(0.1)

    cap.release()


def main() -> None:
    os.makedirs(SENSES_DIR, exist_ok=True)
    print(f"Avatar capture agent — writing to {SENSES_DIR}")
    print("Press Ctrl+C to stop.")

    stop_event = threading.Event()

    audio_thread = threading.Thread(target=audio_loop, args=(stop_event,), daemon=True)
    vision_thread = threading.Thread(target=vision_loop, args=(stop_event,), daemon=True)

    audio_thread.start()
    vision_thread.start()

    try:
        while True:
            has_audio = os.path.exists(os.path.join(SENSES_DIR, "audio_latest.npy"))
            has_video = os.path.exists(os.path.join(SENSES_DIR, "frame_latest.jpg"))
            _write_meta(has_audio, has_video)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nCapture agent stopping...")
        stop_event.set()
        audio_thread.join(timeout=5)
        vision_thread.join(timeout=5)


if __name__ == "__main__":
    main()
