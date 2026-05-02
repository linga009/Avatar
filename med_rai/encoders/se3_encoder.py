import torch
import torch.nn as nn
from med_rai.utils.se3_utils import se3_log


class SE3Encoder(nn.Module):
    """Encodes SE(3) pose via log-map + MLP."""

    def __init__(self, d_out: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.SiLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """R: (B,3,3), t: (B,3) -> (B, d_out)"""
        xi = se3_log(R, t)
        return self.mlp(xi)
