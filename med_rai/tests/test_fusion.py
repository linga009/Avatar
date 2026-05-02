import torch
import pytest
from med_rai.fusion.cross_modal_alignment import CrossModalAlignment

B, SEQ, D = 2, 16, 512

def test_output_shape():
    cma = CrossModalAlignment(d_model=D, n_heads=8)
    v = torch.randn(B, 1, D)
    k = torch.randn(B, 1, D)
    f = torch.randn(B, 1, D)
    l = torch.randn(B, SEQ, D)
    out = cma(v, k, f, l)
    assert out.shape == (B, SEQ + 3, D), f"Expected ({B},{SEQ+3},{D}), got {out.shape}"

def test_danger_signal_amplification():
    """High F/T norm + low-confidence visual should not collapse to zero."""
    cma = CrossModalAlignment(d_model=D, n_heads=8)
    v = torch.zeros(1, 1, D)
    k = torch.randn(1, 1, D)
    f = torch.ones(1, 1, D) * 10.0
    l = torch.randn(1, 4, D)
    out = cma(v, k, f, l)
    assert out.norm() > 0, "Output collapsed to zero — cross-attention broken"
