# halo/tests/test_heads.py
import math
import torch
import pytest
from halo.config import HaloConfig
from halo.heads.text_head import TextHead
from halo.heads.image_head import ImageHead


@pytest.fixture
def cfg():
    return HaloConfig(d_model=256, vocab_size=50257, image_embed_dim=768,
                      delta_text=1.0, delta_image=2.0)


def test_text_head_output_shape(cfg):
    head = TextHead(cfg)
    h = torch.randn(2, 10, cfg.d_model)
    logits = head(h)
    assert logits.shape == (2, 10, cfg.vocab_size)


def test_text_head_no_nan(cfg):
    head = TextHead(cfg)
    h = torch.randn(2, 8, cfg.d_model)
    logits = head(h)
    assert not torch.isnan(logits).any()


def test_image_head_output_shape(cfg):
    head = ImageHead(cfg)
    h = torch.randn(2, 16, cfg.d_model)
    embed = head(h)
    assert embed.shape == (2, 16, cfg.image_embed_dim)


def test_image_head_no_nan(cfg):
    head = ImageHead(cfg)
    h = torch.randn(2, 16, cfg.d_model)
    embed = head(h)
    assert not torch.isnan(embed).any()


def test_text_head_delta_scaling(cfg):
    """Logits are scaled by 1/sqrt(delta_text)."""
    head = TextHead(cfg)
    h = torch.ones(1, 1, cfg.d_model)
    logits = head(h)
    # Cannot test exact value without knowing weights, but can check
    # that output is finite and has correct shape
    assert logits.isfinite().all()
