import torch
import torch.nn as nn


class FTEncoder(nn.Module):
    """MLP encoder for 6-DOF force/torque signals."""

    def __init__(self, d_out: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, 128),
            nn.SiLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, ft: torch.Tensor) -> torch.Tensor:
        """ft: (B, 6) -> (B, d_out)"""
        return self.mlp(ft)
