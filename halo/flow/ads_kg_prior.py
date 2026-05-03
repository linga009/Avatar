# halo/flow/ads_kg_prior.py
import math
import torch
import torch.nn as nn
from halo.config import HaloConfig
from halo.attention.holo_attention import holo_kernel


class AdSKGPrior(nn.Module):
    """AdS Klein-Gordon flow prior for flow matching.

    Given noise x_0 and data x_1, computes the AdS-KG-weighted
    vector field at interpolation time t:

        z_t      = 1 - t                           (depth at time t)
        x_t      = (1-t)*x_0 + t*x_1              (linear interpolation)
        target_v = x_1 - x_0                       (straight-line OT target)
        K        = K_Delta(z_t, x_t; x_0)         (AdS kernel weights)
        v_KG     = softmax_k(K) @ target_v         (KG-weighted vector field)

    The neural network learns only the residual: v = v_KG + eps_theta(x_t, t).
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.log_delta = nn.Parameter(torch.tensor(math.log(cfg.delta_flow)))

    def forward(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the AdS-KG prior vector field at time t.

        Args:
            x_noise: (B, N, d_boundary) noise samples x_0
            x_data:  (B, N, d_boundary) data samples x_1
            t:       (B, 1, 1) interpolation time in [0, 1]
        Returns:
            v_kg:    (B, N, d_boundary) prior vector field
        """
        delta = torch.exp(self.log_delta)
        z_t = 1.0 - t                                   # (B, 1, 1) depth
        x_t = (1.0 - t) * x_noise + t * x_data         # (B, N, d_b)

        # z_t expanded to (B, N, 1) for the kernel
        z_expanded = z_t.expand(x_t.shape[0], x_t.shape[1], 1)

        # K: (B, N, N) — query is x_t, keys are x_noise positions
        K = holo_kernel(z_expanded, x_t, x_noise, delta)  # (B, N, N)
        K_norm = K / (K.sum(dim=-1, keepdim=True) + 1e-8) # (B, N, N)

        # Target: straight-line OT vector field
        target_v = x_data - x_noise                     # (B, N, d_b)

        # AdS-KG weighted vector field
        v_kg = torch.bmm(K_norm, target_v)              # (B, N, d_b)
        return v_kg

    def interpolate(
        self,
        x_noise: torch.Tensor,
        x_data: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_t, v_kg) — interpolated point and prior vector field."""
        x_t = (1.0 - t) * x_noise + t * x_data
        v_kg = self(x_noise, x_data, t)
        return x_t, v_kg
