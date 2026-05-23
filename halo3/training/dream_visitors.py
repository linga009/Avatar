"""Dream Visitors — Whisper & Kokoro as sleep teachers.

During dreaming (CPU-only, GPU free), these models:
  1. Whisper replays stored audio episodes and transcribes them
  2. Kokoro narrates Avatar's discoveries in natural speech

The enriched (audio, text) pairs are saved to disk for Phase 5c GPU training.
Models load, teach, and unload — never present during waking life.

Biological analogy: hippocampal replay during REM sleep enriches memories
with new understanding and novel variations.
"""
from __future__ import annotations
import gc
import json
import logging
import os
import numpy as np

log = logging.getLogger(__name__)


def generate_whisper_pairs(
    data_dir: str = "data",
    max_pairs: int = 10,
) -> list[tuple[np.ndarray, str]]:
    """Phase 5a: Whisper transcribes archived audio episodes.

    Returns list of (audio_waveform, transcribed_text) pairs.
    Loads and unloads Whisper model within this function.
    """
    from halo3.senses.sense_buffer import SenseBuffer
    archive = SenseBuffer.load_audio_archive(data_dir)
    if not archive:
        log.info("Dream visitors: no audio archive — skipping Whisper replay")
        return []

    # Take most recent snapshots
    archive = archive[-max_pairs:]

    try:
        from faster_whisper import WhisperModel
        log.info(f"Dream visitors: loading Whisper tiny (CPU) for {len(archive)} clips...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as e:
        log.warning(f"Dream visitors: Whisper unavailable ({e})")
        return []

    pairs = []
    for filename, audio in archive:
        try:
            segments, _ = model.transcribe(audio, beam_size=1, language="en", vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text and len(text) > 5:
                pairs.append((audio, text))
                log.debug(f"  Whisper: {filename} → '{text[:50]}'")
        except Exception as e:
            log.debug(f"  Whisper: {filename} failed: {e}")

    # Unload model
    del model
    gc.collect()
    log.info(f"Dream visitors: Whisper produced {len(pairs)} pairs, model unloaded")
    return pairs


def generate_kokoro_pairs(
    narrative: list[str],
    max_pairs: int = 10,
    sample_rate: int = 16000,
    duration_samples: int = 32000,
) -> list[tuple[np.ndarray, str]]:
    """Phase 5b: Kokoro narrates Avatar's discoveries in natural speech.

    Takes narrative fragments (discoveries, reflections) and synthesizes
    audio for each. Returns list of (synthesized_audio, text) pairs.
    Loads and unloads Kokoro model within this function.
    """
    # Extract discovery/reflection texts from narrative
    texts = []
    for entry in reversed(narrative):
        if any(kw in entry for kw in ("Discover", "Meta:", "insight", "Dream")):
            # Strip tick prefix like "[Tick 123] "
            clean = entry.split("] ", 1)[-1] if "] " in entry else entry
            if len(clean) > 10:
                texts.append(clean[:200])
        if len(texts) >= max_pairs:
            break

    if not texts:
        log.info("Dream visitors: no narrative texts — skipping Kokoro synthesis")
        return []

    kokoro_dir = os.path.join("data", "kokoro")
    model_path = os.path.join(kokoro_dir, "kokoro-v1.0.onnx")
    voices_path = os.path.join(kokoro_dir, "voices-v1.0.bin")

    try:
        from kokoro_onnx import Kokoro

        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            os.makedirs(kokoro_dir, exist_ok=True)
            log.info("Dream visitors: downloading Kokoro model (~80MB)...")
            import urllib.request
            base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(f"{base}/kokoro-v1.0.onnx", model_path)
            if not os.path.exists(voices_path):
                urllib.request.urlretrieve(f"{base}/voices-v1.0.bin", voices_path)

        log.info(f"Dream visitors: loading Kokoro (CPU) for {len(texts)} texts...")
        kokoro = Kokoro(model_path, voices_path)
    except Exception as e:
        log.warning(f"Dream visitors: Kokoro unavailable ({e})")
        return []

    pairs = []
    for text in texts:
        try:
            audio, sr = kokoro.create(text, voice="af_heart", speed=1.0)
            # Resample to target sample rate
            if sr != sample_rate:
                ratio = sample_rate / sr
                new_len = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
                audio = audio[indices]
            # Pad or trim
            if len(audio) >= duration_samples:
                audio = audio[:duration_samples]
            else:
                audio = np.pad(audio, (0, duration_samples - len(audio)))
            pairs.append((audio.astype(np.float32), text))
            log.debug(f"  Kokoro: synthesized '{text[:40]}...'")
        except Exception as e:
            log.debug(f"  Kokoro synthesis failed: {e}")

    # Unload model
    del kokoro
    gc.collect()
    log.info(f"Dream visitors: Kokoro produced {len(pairs)} pairs, model unloaded")
    return pairs


def save_dream_pairs(
    pairs: list[tuple[np.ndarray, str]],
    output_path: str = "data/dream_training/visitor_pairs.npz",
) -> int:
    """Save (audio, text) pairs to disk for GPU training subprocess."""
    if not pairs:
        return 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audios = np.stack([p[0] for p in pairs])
    texts = [p[1] for p in pairs]
    np.savez(output_path, audios=audios, texts=json.dumps(texts))
    log.info(f"Dream visitors: saved {len(pairs)} pairs to {output_path}")
    return len(pairs)


def run_dream_visitors(
    narrative: list[str],
    data_dir: str = "data",
    max_whisper: int = 10,
    max_kokoro: int = 10,
) -> int:
    """Run Phase 5a + 5b: generate enriched pairs on CPU.

    Returns total number of pairs saved.
    """
    log.info("Dream visitors: Phase 5a — Whisper dream replay...")
    whisper_pairs = generate_whisper_pairs(data_dir, max_whisper)

    log.info("Dream visitors: Phase 5b — Kokoro dream imagination...")
    kokoro_pairs = generate_kokoro_pairs(narrative, max_kokoro)

    all_pairs = whisper_pairs + kokoro_pairs
    if not all_pairs:
        log.info("Dream visitors: no pairs generated — skipping Phase 5c")
        return 0

    output_path = os.path.join(data_dir, "dream_training", "visitor_pairs.npz")
    return save_dream_pairs(all_pairs, output_path)
