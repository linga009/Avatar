import torch
import pytest
from unittest.mock import patch, MagicMock
from med_rai.model import MedRAI

B, SEQ_L, H = 1, 8, 10

def make_batch():
    # Need rotation matrices — generate via simple method without pytorch3d
    import numpy as np
    # Random rotation via QR decomposition
    M = torch.randn(B, 3, 3)
    R = torch.linalg.qr(M)[0]  # (B, 3, 3) orthogonal matrix
    return {
        "images":         torch.randn(B, 3, 224, 224),
        "R":              R,
        "t":              torch.randn(B, 3) * 0.05,
        "ft":             torch.randn(B, 6),
        "token_ids":      torch.randint(0, 1000, (B, SEQ_L)),
        "xi_traj":        torch.randn(B, H, 6),
        "gesture_labels": torch.randint(0, 15, (B,)),
        "text_targets":   torch.randint(0, 65536, (B, SEQ_L + 3)),
    }

@pytest.fixture(scope="module")
def model():
    with patch("med_rai.model.JambaBackbone") as MockBackbone:
        mock_bb = MagicMock()
        mock_bb.return_value = torch.randn(B, SEQ_L + 3, 4096)
        MockBackbone.return_value = mock_bb
        m = MedRAI()
    return m

def test_forward_output_keys(model):
    batch = make_batch()
    out = model(batch)
    assert "v_theta" in out
    assert "text_logits" in out
    assert "gesture_logits" in out

def test_forward_shapes(model):
    batch = make_batch()
    out = model(batch)
    assert out["v_theta"].shape == (B, H, 6)
    assert out["gesture_logits"].shape == (B, 15)

def test_danger_override(model):
    """If Text Head emits WARNING token, v_theta must be zeroed."""
    batch = make_batch()
    with patch.object(model, "_check_danger", return_value=True):
        out = model(batch)
    assert out["v_theta"].abs().sum() == 0, "Trajectory not zeroed on WARNING"
    assert out["forced_gesture"] == "RETRACT"
