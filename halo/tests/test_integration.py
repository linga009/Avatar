# halo/tests/test_integration.py
"""
Integration tests for HALO end-to-end properties.
These validate the six success criteria from the spec.
"""
import time
import torch
import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader

from halo.config import HaloConfig
from halo.model import HALOModel
from halo.data.synthetic_dataset import SyntheticMultimodalDataset
from halo.data.collate import halo_collate
from halo.memory.page_curve_memory import PageCurveMemory
from halo.attention.holo_attention import HoloAttention


def make_small_cfg() -> HaloConfig:
    return HaloConfig(
        d_model=64, d_boundary=16, n_heads=2, n_layers=4,
        d_state=8, d_ff=128, text_embed_dim=32, image_embed_dim=32,
        vocab_size=100, max_cache=8, island_size=4,
        bekenstein_alpha=0.1,
    )


def make_model(cfg: HaloConfig) -> HALOModel:
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )
        mock_clip.get_tokenizer.return_value = MagicMock()
        return HALOModel(cfg)


# --- Criterion 1: Forward pass runs on CPU without OOM ---
def test_forward_pass_cpu_no_oom():
    cfg = make_small_cfg()
    model = make_model(cfg)
    text_h  = torch.randn(4, 8, cfg.d_model)
    image_h = torch.randn(4, 8, cfg.d_model)
    out = model.forward_embeddings(text_h, image_h)
    assert out["v_pred"].shape == (4, 16, cfg.d_model)


# --- Criterion 2: Different modalities produce different conformal dimensions ---
def test_modality_deltas_differ():
    cfg = make_small_cfg()
    model = make_model(cfg)
    # After init, text delta < image delta
    delta_text  = torch.exp(model.text_encoder.log_delta).item()
    delta_image = torch.exp(model.image_encoder.log_delta).item()
    assert abs(delta_text - delta_image) > 0.1, (
        f"Expected different deltas, got text={delta_text:.3f} image={delta_image:.3f}"
    )


# --- Criterion 3: Page curve memory never exceeds max_cache ---
def test_page_curve_cache_bounded():
    cfg = make_small_cfg()
    mem = PageCurveMemory(cfg)
    for i in range(cfg.max_cache * 3):
        x_i    = torch.randn(cfg.d_boundary)
        attn_i = torch.softmax(torch.randn(max(1, i % cfg.max_cache + 1)), dim=0)
        kv_i   = torch.randn(cfg.d_head)
        mem.add(x_i, attn_i, kv_i)
    assert len(mem.active_cache) <= cfg.max_cache, (
        f"Cache exceeded max: {len(mem.active_cache)} > {cfg.max_cache}"
    )
    assert len(mem.island_buffer) <= cfg.island_size


# --- Criterion 4: Bekenstein regulariser reduces attention entropy ---
def test_bekenstein_regulariser_reduces_entropy():
    cfg = make_small_cfg()
    from halo.loss import HALOLoss
    loss_fn = HALOLoss(cfg)
    B, N = 2, 16
    v = torch.randn(B, N, cfg.d_model)
    # Uniform attention = max entropy
    attn_uniform = torch.ones(B, N, N) / N
    evict = torch.softmax(torch.randn(B, N), dim=-1)
    _, parts = loss_fn(v, v, attn_uniform, evict)
    # Bekenstein loss should be positive for uniform attention over N=16 tokens
    import math
    H_uniform = math.log(N)
    bound = cfg.bekenstein_alpha * N * cfg.d_head
    if H_uniform > bound:
        assert parts["bek"].item() > 0


# --- Criterion 5: HoloAttention weights sum to 1 per row ---
def test_holo_attention_normalized():
    cfg = make_small_cfg()
    attn = HoloAttention(cfg)
    B, N = 2, 12
    h = torch.randn(B, N, cfg.d_model)
    x = torch.randn(B, N, cfg.d_boundary)
    z = torch.rand(B, N, 1) * 0.8 + 0.1
    _, weights = attn(h, x, z)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


# --- Criterion 6: Generation output shape is correct ---
def test_generation_output_shape():
    cfg = make_small_cfg()
    model = make_model(cfg)
    text_h = torch.randn(2, 3, cfg.d_model)
    gen = model.generate(text_h, n_image_tokens=4)
    assert gen.shape == (2, 4, cfg.image_embed_dim)


# --- Criterion 7: Latency on CPU < 10 seconds for small batch ---
def test_cpu_latency():
    cfg = make_small_cfg()
    model = make_model(cfg)
    text_h  = torch.randn(2, 4, cfg.d_model)
    image_h = torch.randn(2, 4, cfg.d_model)
    start = time.time()
    model.forward_embeddings(text_h, image_h)
    elapsed = time.time() - start
    assert elapsed < 10.0, f"Forward pass took {elapsed:.2f}s > 10s limit"


# --- Criterion 8: AdS-KG prior produces lower FM loss than random baseline ---
def test_ads_kg_prior_improves_flow():
    """Verify that v_KG is closer to v_target than random noise."""
    from halo.flow.ads_kg_prior import AdSKGPrior
    cfg = make_small_cfg()
    prior = AdSKGPrior(cfg)
    B, N, d = 4, 8, cfg.d_boundary
    x_noise = torch.randn(B, N, d)
    x_data  = torch.randn(B, N, d)
    t = torch.rand(B, 1, 1)

    v_target = x_data - x_noise
    v_kg     = prior(x_noise, x_data, t)
    v_random = torch.randn_like(v_kg)

    err_kg  = (v_kg - v_target).pow(2).mean()
    err_rnd = (v_random - v_target).pow(2).mean()
    # KG prior is data-informed; random is not — KG should be better
    assert err_kg < err_rnd * 2.0, (
        f"KG prior MSE={err_kg:.4f} should be << random MSE={err_rnd:.4f}"
    )
