# halo/backbone/simple_ssm.py
import torch
import torch.nn as nn
from halo.config import HaloConfig


class SimpleSSM(nn.Module):
    """Diagonal-state linear SSM — simplified Mamba without selective scan.

    Recurrence:
        h_t = exp(A) * h_{t-1} + B(x_t)
        y_t = C(h_t) + D * x_t

    A is a learnable diagonal of d_state values (real-valued; stability
    enforced by initialising negative so exp(A) < 1).
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.d_state = cfg.d_state
        # Init A negative so exp(A) < 1 (stable recurrence)
        self.A = nn.Parameter(-torch.ones(cfg.d_state))
        self.B = nn.Linear(cfg.d_model, cfg.d_state, bias=False)
        self.C = nn.Linear(cfg.d_state, cfg.d_model, bias=False)
        self.D = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model)
        Returns:
            y: (B, N, d_model)
        """
        B, N, _ = x.shape
        decay = torch.exp(self.A)          # (d_state,) in (0, 1)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(N):
            h = decay * h + self.B(x[:, t])   # (B, d_state)
            outs.append(self.C(h))             # (B, d_model)
        y = torch.stack(outs, dim=1)           # (B, N, d_model)
        return y + self.D * x                  # skip connection
