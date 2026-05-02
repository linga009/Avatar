import torch
import torch.nn as nn
import math


class _RFMLayer(nn.Module):
    """Cross-attention + feedforward block."""

    def __init__(self, d_hidden: int, n_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 4),
            nn.GELU(),
            nn.Linear(d_hidden * 4, d_hidden),
        )
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """x: (B, H, d_hidden), kv: (B, S, d_hidden)"""
        attn_out, _ = self.cross_attn(x, kv, kv)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class RFMPolicyHead(nn.Module):
    """
    Riemannian Flow Matching head on SE(3).

    Predicts velocity field v_theta(xi_t, t, h) in se(3):
    1. Project noisy trajectory xi_t (B, H, 6) -> (B, H, d_hidden)
    2. Add sinusoidal timestep embedding
    3. Cross-attend to Jamba hidden states h (B, S, d_jamba)
    4. Project back to (B, H, 6)

    Training loss: ||v_theta - (xi_1 - xi_0)||^2
    Inference: 4-step Euler ODE from xi_0 ~ N(0, I)
    """

    def __init__(self, d_jamba: int = 4096, d_hidden: int = 512,
                 n_heads: int = 8, n_layers: int = 4, traj_horizon: int = 10):
        super().__init__()
        self.traj_horizon = traj_horizon
        self.xi_proj = nn.Linear(6, d_hidden)
        self.t_embed = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.h_proj = nn.Linear(d_jamba, d_hidden)
        self.layers = nn.ModuleList([_RFMLayer(d_hidden, n_heads) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_hidden, 6)

    def _sinusoidal_embed(self, t: torch.Tensor, d: int) -> torch.Tensor:
        """t: (B,) -> (B, d) sinusoidal embedding."""
        half = d // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freq[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, xi_t: torch.Tensor, t: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        """
        xi_t: (B, H, 6) noisy se(3) trajectory at timestep t
        t:    (B,)      timestep in [0, 1]
        h:    (B, S, d_jamba) Jamba hidden states
        Returns v_theta: (B, H, 6)
        """
        x = self.xi_proj(xi_t)                                         # (B, H, d_hidden)
        t_emb = self.t_embed(self._sinusoidal_embed(t, x.shape[-1]))   # (B, d_hidden)
        x = x + t_emb.unsqueeze(1)                                     # broadcast over H
        kv = self.h_proj(h)                                             # (B, S, d_hidden)
        for layer in self.layers:
            x = layer(x, kv)
        return self.out_proj(x)                                         # (B, H, 6)
