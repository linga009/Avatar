import torch

try:
    from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
    _HAS_PYTORCH3D = True
except ImportError:  # pragma: no cover – fallback for environments without pytorch3d
    _HAS_PYTORCH3D = False

    def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
        """
        Pure-PyTorch Rodrigues' rotation formula.
        axis_angle: (..., 3)  – direction encodes axis, magnitude encodes angle
        Returns: (..., 3, 3)
        """
        angle = axis_angle.norm(dim=-1, keepdim=True)          # (..., 1)
        eps = 1e-8
        safe_angle = angle.clamp(min=eps)
        axis = axis_angle / safe_angle                          # (..., 3) unit vector

        cos_a = torch.cos(safe_angle)                          # (..., 1)
        sin_a = torch.sin(safe_angle)                          # (..., 1)
        one_minus_cos = 1.0 - cos_a                            # (..., 1)

        x = axis[..., 0:1]
        y = axis[..., 1:2]
        z = axis[..., 2:3]

        # Rodrigues matrix elements  (..., 1) each
        R_flat = torch.cat([
            cos_a + x * x * one_minus_cos,
            x * y * one_minus_cos - z * sin_a,
            x * z * one_minus_cos + y * sin_a,
            y * x * one_minus_cos + z * sin_a,
            cos_a + y * y * one_minus_cos,
            y * z * one_minus_cos - x * sin_a,
            z * x * one_minus_cos - y * sin_a,
            z * y * one_minus_cos + x * sin_a,
            cos_a + z * z * one_minus_cos,
        ], dim=-1)  # (..., 9)

        R = R_flat.reshape(*axis_angle.shape[:-1], 3, 3)

        # For near-zero angles the axis is undefined → identity
        identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
        near_zero = (angle < eps).squeeze(-1).unsqueeze(-1).unsqueeze(-1)  # (...,1,1)
        return torch.where(near_zero.expand_as(R), identity.expand_as(R), R)

    def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
        """
        Pure-PyTorch SO(3) log map using the stable atan2-based formula.
        R: (..., 3, 3)
        Returns: (..., 3) axis-angle vector
        """
        # sin(theta) via skew-symmetric part magnitude
        rx = R[..., 2, 1] - R[..., 1, 2]
        ry = R[..., 0, 2] - R[..., 2, 0]
        rz = R[..., 1, 0] - R[..., 0, 1]
        skew = torch.stack([rx, ry, rz], dim=-1)   # (..., 3)
        sin2_theta = (skew * skew).sum(dim=-1) / 4.0  # sin²θ  (...,)

        # cos(theta) from trace
        cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) / 2.0
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # angle via atan2 for numerical stability
        sin_theta = sin2_theta.clamp(min=0.0).sqrt()
        angle = torch.atan2(sin_theta, cos_theta)   # (...,) in [0, pi]

        # axis = skew / (2 sin θ),  handle small angles with a Taylor series
        eps = 1e-7
        # For small angles: sin θ ≈ θ, coefficient 1/(2sinθ) ≈ 1/(2θ) → axis·θ = skew/2
        # For large angles: standard formula
        # Taylor series: θ/(2·sin θ) ≈ 1/2 + θ²/12 for small θ (avoids division by zero)
        coeff = torch.where(
            sin_theta > eps,
            angle / (2.0 * sin_theta),
            0.5 + angle * angle / 12.0,
        )
        axis_angle = coeff.unsqueeze(-1) * skew   # (..., 3)  = axis * θ

        return axis_angle


def se3_log(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Map SE(3) pose to se(3) Lie algebra 6-vector.
    R: (B, 3, 3) rotation matrix
    t: (B, 3) translation
    Returns xi: (B, 6) where xi[:, :3]=omega (axis-angle), xi[:, 3:]=t
    """
    omega = matrix_to_axis_angle(R)  # (B, 3)
    return torch.cat([omega, t], dim=-1)


def se3_exp(xi: torch.Tensor):
    """
    Map se(3) 6-vector to SE(3) pose.
    xi: (B, 6)
    Returns R: (B, 3, 3), t: (B, 3)

    NOTE: Simplified exponential map — translation is returned unchanged (no V-matrix
    applied). Valid for small perturbations as used in Flow Matching interpolation.
    For full SE(3) exp map, apply: t_out = V(omega) @ t_in.
    """
    omega, t = xi[..., :3], xi[..., 3:]
    R = axis_angle_to_matrix(omega)
    return R, t


def geodesic_distance(R1: torch.Tensor, t1: torch.Tensor,
                      R2: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance between two SE(3) poses.
    Returns scalar distance per batch element: (B,)
    """
    R_rel = R1.transpose(-1, -2) @ R2          # (B, 3, 3)
    omega_rel = matrix_to_axis_angle(R_rel)     # (B, 3)
    rot_dist = omega_rel.norm(dim=-1)           # (B,)
    t_dist = (t1 - t2).norm(dim=-1)             # (B,)
    return rot_dist + t_dist


def clamp_to_workspace(t: torch.Tensor, workspace: dict) -> torch.Tensor:
    """
    Clamp translation to surgical workspace bounds.
    workspace: {"x": (min, max), "y": (min, max), "z": (min, max)}
    t: (B, 3)
    """
    t = t.clone()
    for i, key in enumerate(["x", "y", "z"]):
        lo, hi = workspace[key]
        t[:, i] = t[:, i].clamp(lo, hi)
    return t
