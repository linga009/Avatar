"""Test TTS self-narration pipeline."""
import numpy as np
import pytest


def test_extract_narration_text():
    from halo3.senses.tts_narration import extract_narration_text
    texts = [
        "This is a long document about quantum computing and error correction methods.",
        "Another document with more content."
    ]
    result = extract_narration_text(texts, max_words=10)
    words = result.split()
    assert len(words) <= 10
    assert len(words) > 0


def test_extract_empty_texts():
    from halo3.senses.tts_narration import extract_narration_text
    assert extract_narration_text([], max_words=10) == ""


def test_narrator_init():
    from halo3.senses.tts_narration import TTSNarrator
    tts = TTSNarrator(mode="espeak", sample_rate=16000, duration_samples=32000)
    # available depends on whether espeak-ng is installed
    assert isinstance(tts.available, bool)


def test_narrator_empty_text_returns_silence():
    from halo3.senses.tts_narration import TTSNarrator
    tts = TTSNarrator(mode="espeak", sample_rate=16000, duration_samples=32000)
    audio = tts.narrate("")
    assert audio.shape == (32000,)
    assert np.allclose(audio, 0)


def test_narrator_unavailable_returns_silence():
    from halo3.senses.tts_narration import TTSNarrator
    tts = TTSNarrator(mode="nonexistent", sample_rate=16000, duration_samples=32000)
    assert not tts.available
    audio = tts.narrate("hello world")
    assert audio.shape == (32000,)
    assert np.allclose(audio, 0)
