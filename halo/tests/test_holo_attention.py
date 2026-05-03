# halo/tests/test_holo_attention.py
import torch
import math
import pytest
from halo.config import HaloConfig
from halo.attention.holo_attention import holo_kernel, HoloAttention


@pytest.fixture
def cfg():
    return HaloConfig(d_model=256, d_boundary=64, n_heads=4, d_head=64)


def test_kernel_shape():
    z_i = torch.ones(2, 5, 1) * 0.5
    x_i = torch.randn(2, 5, 64)
    x_j = torch.randn(2, 7, 64)
    K = holo_kernel(z_i, x_i, x_j, delta=1.0)
    assert K.shape == (2, 5, 7)


def test_kernel_positive():
    z_i = torch.rand(2, 4, 1) * 0.9 + 0.05
    x_i = torch.randn(2, 4, 32)
    x_j = torch.randn(2, 4, 32)
    K = holo_kernel(z_i, x_i, x_j, delta=1.5)
    assert (K > 0).all()


def test_kernel_self_is_maximum():
    # Self-attention (x_i == x_j) maximises kernel
    z_i = torch.ones(1, 3, 1) * 0.3
    x = torch.randn(1, 3, 16)
    K_self = holo_kernel(z_i, x, x, delta=2.0)  # (1, 3, 3)
    diag = K_self.diagonal(dim1=1, dim2=2)       # (1, 3)
    off_diag_max = K_self.clone()
    for i in range(3):
        off_diag_max[:, i, i] = 0
    assert (diag >= off_diag_max.max(dim=-1).values).all()


def test_attention_output_shape(cfg):
    attn = HoloAttention(cfg)
    h = torch.randn(2, 10, cfg.d_model)
    x = torch.randn(2, 10, cfg.d_boundary)
    z = torch.rand(2, 10, 1) * 0.9 + 0.05
    out, attn_weights = attn(h, x, z)
    assert out.shape == (2, 10, cfg.d_model)
    assert attn_weights.shape == (2, 10, 10)


def test_attention_weights_sum_to_one(cfg):
    attn = HoloAttention(cfg)
    h = torch.randn(2, 6, cfg.d_model)
    x = torch.randn(2, 6, cfg.d_boundary)
    z = torch.rand(2, 6, 1) * 0.5 + 0.1
    _, w = attn(h, x, z)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_delta_positive(cfg):
    attn = HoloAttention(cfg)
    delta = torch.exp(attn.log_delta)
    assert (delta > 0).all()


def test_no_nan(cfg):
    attn = HoloAttention(cfg)
    h = torch.randn(2, 8, cfg.d_model)
    x = torch.randn(2, 8, cfg.d_boundary)
    z = torch.rand(2, 8, 1) * 0.8 + 0.1
    out, w = attn(h, x, z)
    assert not torch.isnan(out).any()
    assert not torch.isnan(w).any()
