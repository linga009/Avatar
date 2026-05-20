"""AudioSense unit tests — uses mocked Wav2Vec2 to avoid download."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def make_mock_wav2vec2():
    """Return a callable that mimics Wav2Vec2Model output."""
    class FakeOutput:
        def __init__(self):
            # T ≈ 99 frames for 2s at 16kHz
            self.last_hidden_state = np.zeros((1, 99, 768), dtype=np.float32)

    model = MagicMock()
    model.return_value = FakeOutput()
    return model


def make_mock_processor():
    processor = MagicMock()
    processor.return_value = {"input_values": np.zeros((1, 32000), dtype=np.float32)}
    return processor


@patch("halo3.senses.audio_sense.Wav2Vec2Processor.from_pretrained", return_value=make_mock_processor())
@patch("halo3.senses.audio_sense.Wav2Vec2Model.from_pretrained", return_value=make_mock_wav2vec2())
def test_audio_sense_shape(mock_model, mock_proc):
    from halo3.senses.audio_sense import AudioSense
    sense = AudioSense.__new__(AudioSense)
    sense._processor = mock_proc.return_value
    sense._model = mock_model.return_value
    sense._model.eval = MagicMock()

    audio = np.zeros(32000, dtype=np.float32)
    result = sense._encode(audio)

    assert result.shape == (8, 768), f"Expected (8, 768), got {result.shape}"
    assert result.dtype == np.float32


@patch("halo3.senses.audio_sense.Wav2Vec2Processor.from_pretrained", return_value=make_mock_processor())
@patch("halo3.senses.audio_sense.Wav2Vec2Model.from_pretrained", return_value=make_mock_wav2vec2())
def test_audio_sense_zero_input_safe(mock_model, mock_proc):
    """Zero audio should not raise and should return finite values."""
    from halo3.senses.audio_sense import AudioSense
    sense = AudioSense.__new__(AudioSense)
    sense._processor = mock_proc.return_value
    sense._model = mock_model.return_value
    sense._model.eval = MagicMock()

    audio = np.zeros(32000, dtype=np.float32)
    result = sense._encode(audio)
    assert np.all(np.isfinite(result))
