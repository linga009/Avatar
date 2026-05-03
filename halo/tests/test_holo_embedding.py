# halo/tests/test_holo_embedding.py
import torch
import pytest
from halo.config import HaloConfig
from halo.embeddings.holo_embedding import HoloEmbedding


@pytest.fixture
def cfg():
    return HaloConfig(d_model=256, d_boundary=64)


def test_output_shapes(cfg):
    embed = HoloEmbedding(cfg)
    h = torch.randn(2, 10, cfg.d_model)
    x, z = embed(h)
    assert x.shape == (2, 10, cfg.d_boundary)
    assert z.shape == (2, 10, 1)


def test_z_in_unit_interval(cfg):
    embed = HoloEmbedding(cfg)
    h = torch.randn(4, 20, cfg.d_model)
    _, z = embed(h)
    assert (z > 0).all(), "z must be positive"
    assert (z < 1).all(), "z must be < 1"


def test_no_nan(cfg):
    embed = HoloEmbedding(cfg)
    h = torch.randn(2, 8, cfg.d_model)
    x, z = embed(h)
    assert not torch.isnan(x).any()
    assert not torch.isnan(z).any()


def test_different_tokens_different_z(cfg):
    embed = HoloEmbedding(cfg)
    h = torch.randn(1, 5, cfg.d_model)
    _, z = embed(h)
    # z values should differ across positions
    assert z.std() > 1e-6
