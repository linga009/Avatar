# halo/heads/image_head.py
import math
import torch
import torch.nn as nn
from halo.config import HaloConfig


class ImageHead(nn.Module):
    """Image output head — projects d_model -> image_embed_dim (CLIP space).

    Higher conformal dimension (delta_image=2) means sharper, more localised
    image features compared to text (delta_text=1).
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.image_embed_dim, bias=False)
        self.log_delta = nn.Parameter(torch.tensor(math.log(cfg.delta_image)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, d_model)
        Returns:
            embed: (B, N, image_embed_dim)
        """
        delta = torch.exp(self.log_delta)
        return self.proj(h) / (delta.sqrt() + 1e-8)
