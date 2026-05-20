"""VisionSense unit tests — mocked CLIP to avoid download."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


def make_mock_clip_model():
    class FakeOutput:
        pooler_output = np.zeros((1, 512), dtype=np.float32)
    model = MagicMock()
    model.return_value = FakeOutput()
    return model


def make_mock_clip_processor():
    processor = MagicMock()
    processor.return_value = {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
    return processor


@patch("halo3.senses.vision_sense.CLIPProcessor.from_pretrained",
       return_value=make_mock_clip_processor())
@patch("halo3.senses.vision_sense.CLIPVisionModel.from_pretrained",
       return_value=make_mock_clip_model())
def test_vision_sense_shape(mock_model, mock_proc):
    from halo3.senses.vision_sense import VisionSense
    sense = VisionSense.__new__(VisionSense)
    sense._processor = mock_proc.return_value
    sense._model = mock_model.return_value
    sense._model.eval = MagicMock()

    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = sense._encode(img)

    assert result.shape == (512,), f"Expected (512,), got {result.shape}"
    assert result.dtype == np.float32


@patch("halo3.senses.vision_sense.CLIPProcessor.from_pretrained",
       return_value=make_mock_clip_processor())
@patch("halo3.senses.vision_sense.CLIPVisionModel.from_pretrained",
       return_value=make_mock_clip_model())
def test_vision_sense_finite(mock_model, mock_proc):
    from halo3.senses.vision_sense import VisionSense
    sense = VisionSense.__new__(VisionSense)
    sense._processor = mock_proc.return_value
    sense._model = mock_model.return_value
    sense._model.eval = MagicMock()

    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = sense._encode(img)
    assert np.all(np.isfinite(result))
