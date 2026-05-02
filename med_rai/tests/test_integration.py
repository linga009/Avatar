# med_rai/tests/test_integration.py
import torch
import pytest
import time
from unittest.mock import patch, MagicMock
from med_rai.model import MedRAI
from med_rai.heads.text_head import TextHead

B, SEQ_L, H = 1, 8, 10


def make_batch(device="cpu"):
    M = torch.randn(B, 3, 3)
    R = torch.linalg.qr(M)[0]  # (B, 3, 3) orthogonal
    return {
        "images":         torch.randn(B, 3, 224, 224, device=device),
        "R":              R.to(device),
        "t":              torch.randn(B, 3, device=device) * 0.05,
        "ft":             torch.randn(B, 6, device=device),
        "token_ids":      torch.randint(0, 1000, (B, SEQ_L), device=device),
        "xi_traj":        torch.randn(B, H, 6, device=device),
        "gesture_labels": torch.randint(0, 15, (B,), device=device),
        "text_targets":   torch.randint(0, 65536, (B, SEQ_L + 3), device=device),
    }


@pytest.fixture(scope="module")
def model():
    with patch("med_rai.model.JambaBackbone") as MockBackbone:
        mock_bb = MagicMock()
        mock_bb.return_value = torch.randn(B, SEQ_L + 3, 4096)
        MockBackbone.return_value = mock_bb
        m = MedRAI()
    return m


def test_danger_signal_pathway(model):
    """
    Force spike: extreme F/T. Danger override must zero trajectory and set RETRACT.
    """
    batch = make_batch()
    batch["ft"] = torch.ones(B, 6) * 100.0  # extreme force spike

    with patch.object(model, "_check_danger", return_value=True):
        out = model(batch)

    assert out["v_theta"].abs().sum().item() == 0.0, (
        "Trajectory must be zeroed after WARNING signal"
    )
    assert out.get("forced_gesture") == "RETRACT", (
        f"Expected RETRACT, got {out.get('forced_gesture')}"
    )


def test_full_inference_shapes(model):
    """All 4 modalities in -> correct output shapes."""
    batch = make_batch()
    out = model(batch)
    assert out["v_theta"].shape == (B, H, 6)
    assert out["text_logits"].shape[0] == B
    assert out["gesture_logits"].shape == (B, 15)


def test_no_nan_in_outputs(model):
    """No NaN in any output under normal inputs."""
    batch = make_batch()
    out = model(batch)
    assert not torch.isnan(out["v_theta"]).any(), "NaN in v_theta"
    assert not torch.isnan(out["text_logits"]).any(), "NaN in text_logits"
    assert not torch.isnan(out["gesture_logits"]).any(), "NaN in gesture_logits"


def test_trajectory_workspace_bounds(model):
    """After ODE integration, translations must respect workspace after clamping."""
    from med_rai.model import WORKSPACE
    from med_rai.utils.se3_utils import se3_exp
    batch = make_batch()
    out = model(batch)
    _, t_out = se3_exp(out["v_theta"].reshape(-1, 6))
    for i, key in enumerate(["x", "y", "z"]):
        lo, hi = WORKSPACE[key]
        assert (t_out[:, i] >= lo - 1e-4).all(), f"{key} below workspace min {lo}"
        assert (t_out[:, i] <= hi + 1e-4).all(), f"{key} above workspace max {hi}"


def test_gesture_vocabulary_size(model):
    """Gesture logits must have exactly 15 classes."""
    batch = make_batch()
    out = model(batch)
    assert out["gesture_logits"].shape[-1] == 15


def test_latency(model):
    """Single CPU inference pass must complete in < 5s."""
    batch = make_batch()
    start = time.time()
    _ = model(batch)
    elapsed = time.time() - start
    assert elapsed < 5.0, f"Inference took {elapsed:.2f}s — too slow"
