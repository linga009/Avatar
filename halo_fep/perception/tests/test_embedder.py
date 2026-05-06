import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_mock_st_model(dim=384):
    m = MagicMock()
    m.encode.return_value = np.random.randn(1, dim).astype(np.float32)
    return m


def make_mock_clip(dim=512):
    proc = MagicMock()
    model = MagicMock()
    model.get_image_features.return_value = MagicMock(
        detach=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.random.randn(1, dim).astype(np.float32)))
    )
    return proc, model


@patch("halo_fep.perception.embedder.SentenceTransformer")
def test_embed_text_shape(MockST):
    MockST.return_value = make_mock_st_model(384)
    from halo_fep.perception.embedder import Embedder
    emb = Embedder(d_model=256)
    out = emb.embed_text("hello world")
    assert out.shape == (256,)
    assert out.dtype == np.float32


@patch("halo_fep.perception.embedder.SentenceTransformer")
def test_embed_text_normalized(MockST):
    MockST.return_value = make_mock_st_model(384)
    from halo_fep.perception.embedder import Embedder
    emb = Embedder(d_model=256)
    out = emb.embed_text("hello")
    norm = np.linalg.norm(out)
    assert abs(norm - 1.0) < 1e-5


@patch("halo_fep.perception.embedder.SentenceTransformer")
def test_embed_image_none_returns_zero(MockST):
    MockST.return_value = make_mock_st_model(384)
    from halo_fep.perception.embedder import Embedder
    emb = Embedder(d_model=256)
    out = emb.embed_image(None)
    assert out.shape == (256,)
    assert np.all(out == 0.0)
