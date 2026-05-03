# halo/tests/test_modality_encoders.py
import torch
import pytest
from unittest.mock import patch, MagicMock
from halo.config import HaloConfig
from halo.embeddings.modality_encoders import TextEncoder, ImageEncoder


@pytest.fixture
def cfg():
    return HaloConfig(d_model=256, text_embed_dim=768, image_embed_dim=768)


def test_text_encoder_output_shape(cfg):
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )
        mock_clip.get_tokenizer.return_value = mock_tokenizer
        mock_model.encode_text.return_value = torch.randn(2, 768)

        enc = TextEncoder(cfg)
        token_ids = torch.randint(0, 100, (2, 16))
        out = enc(token_ids)
        assert out.shape == (2, cfg.d_model)


def test_image_encoder_output_shape(cfg):
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )
        mock_model.encode_image.return_value = torch.randn(2, 768)

        enc = ImageEncoder(cfg)
        images = torch.randn(2, 3, 224, 224)
        out = enc(images)
        assert out.shape == (2, cfg.d_model)


def test_text_encoder_frozen(cfg):
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )
        mock_clip.get_tokenizer.return_value = MagicMock()

        enc = TextEncoder(cfg)
        for name, p in enc.clip.named_parameters():
            assert not p.requires_grad, f"CLIP param {name} should be frozen"


def test_image_encoder_frozen(cfg):
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )

        enc = ImageEncoder(cfg)
        for name, p in enc.clip.named_parameters():
            assert not p.requires_grad, f"CLIP param {name} should be frozen"


def test_delta_learnable(cfg):
    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_clip.create_model_and_transforms.return_value = (
            MagicMock(), MagicMock(), MagicMock()
        )
        mock_clip.get_tokenizer.return_value = MagicMock()
        enc = TextEncoder(cfg)
        assert enc.log_delta.requires_grad
