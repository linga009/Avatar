# halo/tests/test_loss.py
import torch
import pytest
from halo.config import HaloConfig
from halo.loss import HALOLoss


@pytest.fixture
def cfg():
    return HaloConfig(
        d_model=256, d_head=64, bekenstein_alpha=0.1,
        lambda_bek=0.1, lambda_thermo=0.05, lambda_page=0.05
    )


def test_loss_scalar(cfg):
    loss_fn = HALOLoss(cfg)
    B, N = 2, 8
    v_pred   = torch.randn(B, N, cfg.d_model)
    v_target = torch.randn(B, N, cfg.d_model)
    attn_weights = torch.softmax(torch.randn(B, N, N), dim=-1)
    evict_scores = torch.softmax(torch.randn(B, N), dim=-1)
    total, parts = loss_fn(v_pred, v_target, attn_weights, evict_scores)
    assert total.shape == ()
    assert total.isfinite()


def test_loss_parts_keys(cfg):
    loss_fn = HALOLoss(cfg)
    B, N = 2, 6
    v_pred   = torch.randn(B, N, cfg.d_model)
    v_target = torch.randn(B, N, cfg.d_model)
    attn     = torch.softmax(torch.randn(B, N, N), dim=-1)
    evict    = torch.softmax(torch.randn(B, N), dim=-1)
    _, parts = loss_fn(v_pred, v_target, attn, evict)
    assert "fm" in parts
    assert "bek" in parts
    assert "thermo" in parts
    assert "page" in parts


def test_fm_loss_zero_on_perfect(cfg):
    loss_fn = HALOLoss(cfg)
    v = torch.randn(2, 5, cfg.d_model)
    attn = torch.softmax(torch.randn(2, 5, 5), dim=-1)
    evict = torch.softmax(torch.randn(2, 5), dim=-1)
    _, parts = loss_fn(v, v, attn, evict)
    assert parts["fm"].item() < 1e-6


def test_bekenstein_nonnegative(cfg):
    loss_fn = HALOLoss(cfg)
    v = torch.randn(2, 4, cfg.d_model)
    attn = torch.softmax(torch.randn(2, 4, 4), dim=-1)
    evict = torch.softmax(torch.randn(2, 4), dim=-1)
    _, parts = loss_fn(v, v, attn, evict)
    assert parts["bek"].item() >= 0


def test_no_nan(cfg):
    loss_fn = HALOLoss(cfg)
    v_pred   = torch.randn(2, 8, cfg.d_model)
    v_target = torch.randn(2, 8, cfg.d_model)
    attn     = torch.softmax(torch.randn(2, 8, 8), dim=-1)
    evict    = torch.softmax(torch.randn(2, 8), dim=-1)
    total, parts = loss_fn(v_pred, v_target, attn, evict)
    assert not torch.isnan(total)
    for v in parts.values():
        assert not torch.isnan(v)
