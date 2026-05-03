# halo/embeddings/holo_embedding.py
import torch
import torch.nn as nn
from halo.config import HaloConfig


class HoloEmbedding(nn.Module):
    """Projects token embeddings into Poincaré upper half-space (x, z).

    x ∈ R^d_boundary is the boundary coordinate (position in semantic space).
    z ∈ (0, 1) is the holographic depth:
        z ≈ 1  →  deep in bulk (UV / fine-grained tokens)
        z ≈ 0  →  near boundary (IR / coarse semantic tokens)
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.x_proj = nn.Linear(cfg.d_model, cfg.d_boundary)
        self.z_proj = nn.Linear(cfg.d_model, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, N, d_model) token embeddings
        Returns:
            x: (B, N, d_boundary) boundary coordinates
            z: (B, N, 1) depths in (0, 1)
        """
        x = self.x_proj(h)               # (B, N, d_boundary)
        z = torch.sigmoid(self.z_proj(h)) # (B, N, 1)
        return x, z
