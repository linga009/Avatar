"""Windows host capture agent for Avatar senses.

Records microphone + camera and writes to the Docker shared volume.
Run from the worktree root: python capture_agent/capture_agent.py

The Docker volume is ./data/senses/ which maps to /app/data/senses/ in the container.
"""
from __future__ import annotations
import json
import os
import time
import threading
import numpy as np

SENSES_DIR = os.path.join("data", "senses")
SAMPLE_RATE = 16000       # Hz — Wav2Vec2 expects 16kHz
AUDIO_CHUNK_SECS = 2      # seconds of audio per chunk
FRAME_INTERVAL_SECS = 10  # seconds between camera captures
MOTION_THRESHOLD = 30     # pixel diff threshold for motion capture


def _write_meta(has_audio: bool, has_video: bool) -> None:
    meta = {"has_audio": has_audio, "has_video": has_video, "timestamp": time.time()}
    tmp = os.path.join(SENSES_DIR, "meta_tmp.json")
    final = os.path.join(SENSES_DIR, "meta.json")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, final)  # atomic write


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
            tmp_path = audio_path.replace(".npy", "_tmp")
            np.save(tmp_path, audio_mono)  # np.save appends .npy automatically
            os.replace(tmp_path + ".npy", audio_path)
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
            tmp = frame_path.replace(".jpg", "_tmp.jpg")
            cv2.imwrite(tmp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            os.replace(tmp, frame_path)
            last_capture = now
            if motion:
                print(f"Vision: motion detected, captured frame")

        prev_gray = gray
        time.sleep(0.1)

    cap.release()


def main() -> None:
    os.makedirs(SENSES_DIR, exist_ok=True)
    print(f"Avatar capture agent — writing to {os.path.abspath(SENSES_DIR)}")
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
