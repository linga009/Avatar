# halo/attention/holo_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from halo.config import HaloConfig


def holo_kernel(
    z_i: torch.Tensor,
    x_i: torch.Tensor,
    x_j: torch.Tensor,
    delta: float | torch.Tensor,
) -> torch.Tensor:
    """AdS bulk-to-boundary propagator kernel.

    K_Δ(z_i, x_i; x_j) = (z_i / (z_i² + ||x_i - x_j||² + ε))^Δ

    Args:
        z_i:  (B, N, 1)   depth of query tokens
        x_i:  (B, N, d_b) boundary coords of query tokens
        x_j:  (B, M, d_b) boundary coords of key tokens
        delta: scalar or broadcastable tensor — conformal dimension Δ > 0
    Returns:
        K:    (B, N, M)   kernel values (all positive)
    """
    # Pairwise squared distance ||x_i - x_j||²
    diff = x_i.unsqueeze(2) - x_j.unsqueeze(1)   # (B, N, M, d_b)
    dist_sq = (diff ** 2).sum(dim=-1)              # (B, N, M)

    z_sq = z_i ** 2                                # (B, N, 1)
    ratio = z_i / (z_sq + dist_sq + 1e-8)         # (B, N, M)
    return ratio ** delta                          # (B, N, M)


class HoloAttention(nn.Module):
    """Multi-head holographic attention using K_Δ as the kernel.

    Each head has a learnable conformal dimension Δ_h = exp(log_delta_h).
    The final kernel averages over heads.
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        # Learnable log(Δ) per head — exp gives Δ > 0
        delta_init = (cfg.delta_text + cfg.delta_image) / 2.0
        self.log_delta = nn.Parameter(
            torch.full((cfg.n_heads,), math.log(delta_init))
        )
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.n_heads = cfg.n_heads

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:    (B, N, d_model) value embeddings
            x:    (B, N, d_boundary) boundary coordinates
            z:    (B, N, 1) depths
            mask: (B, N, N) optional boolean mask (True = ignore)
        Returns:
            out:         (B, N, d_model)
            attn_weights:(B, N, N) averaged over heads
        """
        delta = torch.exp(self.log_delta)  # (n_heads,)

        # Compute kernel for each head and average
        K_heads = torch.stack(
            [holo_kernel(z, x, x, delta[h]) for h in range(self.n_heads)],
            dim=0,
        )  # (n_heads, B, N, N)
        K = K_heads.mean(dim=0)  # (B, N, N)

        if mask is not None:
            K = K.masked_fill(mask, 0.0)

        # Normalize to get attention weights
        attn_weights = K / (K.sum(dim=-1, keepdim=True) + 1e-8)  # (B, N, N)

        # Attend over values
        v = self.v_proj(h)                       # (B, N, d_model)
        ctx = torch.bmm(attn_weights, v)         # (B, N, d_model)
        out = self.out_proj(ctx)                 # (B, N, d_model)
        return out, attn_weights
