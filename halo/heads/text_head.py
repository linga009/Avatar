# halo/heads/text_head.py
import math
import torch
import torch.nn as nn
from halo.config import HaloConfig


class TextHead(nn.Module):
    """Text output head — projects d_model -> vocab_size.

    Logits are scaled by 1/sqrt(delta_text) to encode the conformal dimension:
    higher-delta operators produce sharper distributions (lower temperature).
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.log_delta = nn.Parameter(torch.tensor(math.log(cfg.delta_text)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, d_model)
        Returns:
            logits: (B, N, vocab_size)
        """
        delta = torch.exp(self.log_delta)
        return self.proj(h) / (delta.sqrt() + 1e-8)
