# halo/tests/test_ads_kg_prior.py
import torch
import pytest
from halo.config import HaloConfig
from halo.flow.ads_kg_prior import AdSKGPrior


@pytest.fixture
def cfg():
    return HaloConfig(d_boundary=64, delta_flow=1.5)


def test_prior_output_shape(cfg):
    prior = AdSKGPrior(cfg)
    B, N, d = 2, 8, cfg.d_boundary
    x_noise = torch.randn(B, N, d)
    x_data  = torch.randn(B, N, d)
    t = torch.rand(B, 1, 1)
    v_kg = prior(x_noise, x_data, t)
    assert v_kg.shape == (B, N, d)


def test_prior_no_nan(cfg):
    prior = AdSKGPrior(cfg)
    x_noise = torch.randn(2, 6, cfg.d_boundary)
    x_data  = torch.randn(2, 6, cfg.d_boundary)
    t = torch.rand(2, 1, 1)
    v_kg = prior(x_noise, x_data, t)
    assert not torch.isnan(v_kg).any()


def test_prior_at_t0_is_data_minus_noise(cfg):
    """At t=0, x_t = x_noise, z_t = 1.0 (deep bulk).
    The KG kernel at z=1 is uniform-ish -> v_KG ~= mean(x_data - x_noise).
    """
    prior = AdSKGPrior(cfg)
    B, N, d = 1, 4, cfg.d_boundary
    x_noise = torch.zeros(B, N, d)
    x_data  = torch.ones(B, N, d)
    t = torch.zeros(B, 1, 1)
    v_kg = prior(x_noise, x_data, t)
    # All tokens identical -> v_KG should be all-ones
    assert torch.allclose(v_kg, torch.ones_like(v_kg), atol=0.1)


def test_delta_positive(cfg):
    prior = AdSKGPrior(cfg)
    assert torch.exp(prior.log_delta).item() > 0


def test_interpolated_x_shape(cfg):
    prior = AdSKGPrior(cfg)
    x_noise = torch.randn(3, 5, cfg.d_boundary)
    x_data  = torch.randn(3, 5, cfg.d_boundary)
    t = torch.rand(3, 1, 1)
    x_t, v_kg = prior.interpolate(x_noise, x_data, t)
    assert x_t.shape == x_noise.shape
    assert v_kg.shape == x_noise.shape
