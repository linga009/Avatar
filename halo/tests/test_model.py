# halo/tests/test_model.py
import torch
import pytest
from unittest.mock import patch, MagicMock
from halo.config import HaloConfig
from halo.model import HALOModel


def make_model(cfg: HaloConfig) -> HALOModel:
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )
        mock_clip.get_tokenizer.return_value = MagicMock()
        return HALOModel(cfg)


@pytest.fixture
def cfg():
    return HaloConfig(d_model=64, d_boundary=16, n_heads=2, n_layers=4,
                      d_state=8, d_ff=128, text_embed_dim=32,
                      image_embed_dim=32, vocab_size=100,
                      max_cache=16, island_size=4)


def test_forward_output_keys(cfg):
    model = make_model(cfg)
    text_h  = torch.randn(2, 1, cfg.d_model)
    image_h = torch.randn(2, 1, cfg.d_model)
    out = model.forward_embeddings(text_h, image_h)
    assert "v_pred" in out
    assert "v_target" in out
    assert "attn_weights" in out
    assert "evict_scores" in out


def test_forward_shapes(cfg):
    model = make_model(cfg)
    N_text, N_image = 4, 4
    text_h  = torch.randn(2, N_text,  cfg.d_model)
    image_h = torch.randn(2, N_image, cfg.d_model)
    out = model.forward_embeddings(text_h, image_h)
    N = N_text + N_image
    assert out["v_pred"].shape   == (2, N, cfg.d_model)
    assert out["v_target"].shape == (2, N, cfg.d_model)


def test_no_nan(cfg):
    model = make_model(cfg)
    text_h  = torch.randn(2, 3, cfg.d_model)
    image_h = torch.randn(2, 3, cfg.d_model)
    out = model.forward_embeddings(text_h, image_h)
    for k, v in out.items():
        assert not torch.isnan(v).any(), f"NaN in {k}"


def test_loss_decreases(cfg):
    """After a few gradient steps, total loss should not explode."""
    model = make_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        text_h  = torch.randn(2, 4, cfg.d_model)
        image_h = torch.randn(2, 4, cfg.d_model)
        out = model.forward_embeddings(text_h, image_h)
        total, _ = model.loss_fn(
            out["v_pred"], out["v_target"],
            out["attn_weights"], out["evict_scores"]
        )
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())

    assert losses[-1] < losses[0] * 2, "Loss should not explode"


def test_generate_shape(cfg):
    model = make_model(cfg)
    text_h = torch.randn(1, 2, cfg.d_model)
    gen = model.generate(text_h, n_image_tokens=4)
    assert gen.shape == (1, 4, cfg.image_embed_dim)
