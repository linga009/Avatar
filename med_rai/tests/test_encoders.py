import torch
import pytest
from med_rai.encoders.vision_encoder import SurgicalViTEncoder
from med_rai.encoders.se3_encoder import SE3Encoder
from med_rai.encoders.ft_encoder import FTEncoder
from med_rai.encoders.language_embedder import LanguageEmbedder

D_OUT = 512

def test_vision_encoder_output_shape():
    enc = SurgicalViTEncoder(d_out=D_OUT)
    imgs = torch.randn(2, 3, 224, 224)
    out = enc(imgs)
    assert out.shape == (2, D_OUT), f"Expected (2,{D_OUT}), got {out.shape}"

def test_vision_encoder_frozen():
    enc = SurgicalViTEncoder(d_out=D_OUT)
    vit_params = [p for n, p in enc.named_parameters() if "vit" in n and "probe" not in n]
    assert all(not p.requires_grad for p in vit_params), "ViT backbone must be frozen"

def test_se3_encoder_output_shape():
    enc = SE3Encoder(d_out=D_OUT)
    from med_rai.utils.se3_utils import se3_exp
    xi = torch.randn(2, 6) * 0.1
    R, t = se3_exp(xi)
    out = enc(R, t)
    assert out.shape == (2, D_OUT)

def test_ft_encoder_output_shape():
    enc = FTEncoder(d_out=D_OUT)
    ft = torch.randn(2, 6)
    out = enc(ft)
    assert out.shape == (2, D_OUT)

def test_language_embedder_output_shape():
    emb = LanguageEmbedder(d_out=D_OUT)
    tokens = torch.randint(0, 1000, (2, 16))
    out = emb(tokens)
    assert out.shape == (2, 16, D_OUT), f"Expected (2,16,{D_OUT}), got {out.shape}"
