import torch
import pytest
from med_rai.utils.se3_utils import se3_log, se3_exp, geodesic_distance, clamp_to_workspace


def make_random_se3(batch=4):
    """Return random (R, t): R in SO(3), t in R^3."""
    try:
        from pytorch3d.transforms import random_rotations
        R = random_rotations(batch)          # (B, 3, 3)
    except ImportError:
        # Pure-torch fallback: random rotation via QR decomposition
        M = torch.randn(batch, 3, 3)
        Q, _ = torch.linalg.qr(M)
        # Ensure proper rotation (det = +1)
        det = torch.linalg.det(Q)
        Q = Q * det.unsqueeze(-1).unsqueeze(-1)
        R = Q
    t = torch.randn(batch, 3) * 0.1     # (B, 3) small translations
    return R, t


def test_se3_log_shape():
    R, t = make_random_se3(4)
    xi = se3_log(R, t)
    assert xi.shape == (4, 6), f"Expected (4,6), got {xi.shape}"


def test_se3_exp_shape():
    xi = torch.randn(4, 6) * 0.1
    R, t = se3_exp(xi)
    assert R.shape == (4, 3, 3)
    assert t.shape == (4, 3)


def test_se3_roundtrip():
    """exp(log(T)) ≈ T for random SE(3) poses."""
    R, t = make_random_se3(100)
    xi = se3_log(R, t)
    R_hat, t_hat = se3_exp(xi)
    assert torch.allclose(R, R_hat, atol=1e-5), f"Max R error: {(R - R_hat).abs().max()}"
    assert torch.allclose(t, t_hat, atol=1e-5), f"Max t error: {(t - t_hat).abs().max()}"


def test_geodesic_distance_zero():
    R, t = make_random_se3(4)
    d = geodesic_distance(R, t, R, t)
    assert torch.allclose(d, torch.zeros(4), atol=1e-5)


def test_clamp_to_workspace():
    workspace = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.05, 0.2)}
    t_out = torch.tensor([[0.5, 0.5, 0.5]])   # way outside workspace
    t_clamped = clamp_to_workspace(t_out, workspace)
    assert t_clamped[0, 0] <= 0.1
    assert t_clamped[0, 2] <= 0.2


def test_clamp_to_workspace_preserves_inbounds():
    workspace = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.05, 0.2)}
    t_in = torch.tensor([[0.05, 0.0, 0.1]])   # within bounds
    t_clamped = clamp_to_workspace(t_in, workspace)
    assert torch.allclose(t_in, t_clamped)
