import torch
import pytest
from med_rai.heads.rfm_policy_head import RFMPolicyHead
from med_rai.heads.text_head import TextHead
from med_rai.heads.gesture_head import GestureHead

B, SEQ, D_JAMBA, D_HIDDEN, H = 2, 19, 4096, 512, 10
SE3_DIM = 6

@pytest.fixture
def rfm_head():
    return RFMPolicyHead(d_jamba=D_JAMBA, d_hidden=D_HIDDEN, n_heads=8, n_layers=4)

def test_rfm_output_shape(rfm_head):
    xi_t = torch.randn(B, H, SE3_DIM)
    t = torch.rand(B)
    h = torch.randn(B, SEQ, D_JAMBA)
    v_theta = rfm_head(xi_t, t, h)
    assert v_theta.shape == (B, H, SE3_DIM), f"Expected ({B},{H},{SE3_DIM}), got {v_theta.shape}"

def test_rfm_no_nan(rfm_head):
    xi_t = torch.randn(B, H, SE3_DIM)
    t = torch.rand(B)
    h = torch.randn(B, SEQ, D_JAMBA)
    v_theta = rfm_head(xi_t, t, h)
    assert not torch.isnan(v_theta).any(), "NaN in RFM output"

def test_rfm_inference_ode(rfm_head):
    """4-step Euler ODE produces valid SE(3) pose (no NaN, finite)."""
    from med_rai.utils.se3_utils import se3_exp
    h = torch.randn(1, SEQ, D_JAMBA)
    xi = torch.randn(1, H, SE3_DIM)
    dt = 1.0 / 4
    for step in range(4):
        t = torch.full((1,), step * dt)
        v = rfm_head(xi, t, h)
        xi = xi + dt * v
    R, t_out = se3_exp(xi.reshape(-1, SE3_DIM))
    assert not torch.isnan(R).any()
    assert not torch.isnan(t_out).any()

N_GESTURES = 15

def test_text_head_output_shape():
    head = TextHead(d_jamba=D_JAMBA, vocab_size=65536)
    h = torch.randn(B, SEQ, D_JAMBA)
    logits = head(h)
    assert logits.shape == (B, SEQ, 65536), f"Got {logits.shape}"

def test_text_head_warning_token():
    head = TextHead(d_jamba=D_JAMBA, vocab_size=65536)
    assert hasattr(head, "WARNING_TOKEN_ID"), "TextHead must expose WARNING_TOKEN_ID"
    assert head.WARNING_TOKEN_ID == 1

def test_gesture_head_output_shape():
    head = GestureHead(d_jamba=D_JAMBA, n_gestures=N_GESTURES)
    h = torch.randn(B, SEQ, D_JAMBA)
    logits = head(h)
    assert logits.shape == (B, N_GESTURES), f"Got {logits.shape}"
