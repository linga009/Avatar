# halo/tests/test_backbone.py
import torch
import pytest
from halo.config import HaloConfig
from halo.backbone.simple_ssm import SimpleSSM
from halo.backbone.halo_backbone import HALOBackbone


@pytest.fixture
def cfg():
    return HaloConfig(d_model=256, d_boundary=64, n_heads=4, n_layers=8,
                      d_state=16, d_ff=512)


def test_ssm_output_shape(cfg):
    ssm = SimpleSSM(cfg)
    x = torch.randn(2, 10, cfg.d_model)
    out = ssm(x)
    assert out.shape == (2, 10, cfg.d_model)


def test_ssm_no_nan(cfg):
    ssm = SimpleSSM(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = ssm(x)
    assert not torch.isnan(out).any()


def test_backbone_output_shape(cfg):
    backbone = HALOBackbone(cfg)
    h = torch.randn(2, 12, cfg.d_model)
    x = torch.randn(2, 12, cfg.d_boundary)
    z = torch.rand(2, 12, 1) * 0.8 + 0.1
    out = backbone(h, x, z)
    assert out.shape == (2, 12, cfg.d_model)


def test_backbone_no_nan(cfg):
    backbone = HALOBackbone(cfg)
    h = torch.randn(2, 10, cfg.d_model)
    x = torch.randn(2, 10, cfg.d_boundary)
    z = torch.rand(2, 10, 1) * 0.8 + 0.1
    out = backbone(h, x, z)
    assert not torch.isnan(out).any()


def test_backbone_layer_count(cfg):
    backbone = HALOBackbone(cfg)
    # Pattern [S,S,S,H,S,S,S,H]: 2 HoloAttention in 8 layers
    holo_count = sum(1 for layer in backbone.layers
                     if layer["type"] == "holo")
    ssm_count = sum(1 for layer in backbone.layers
                    if layer["type"] == "ssm")
    assert holo_count == 2
    assert ssm_count == 6
