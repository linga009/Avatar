import torch
import torch.nn as nn


class TextHead(nn.Module):
    """
    LM head projecting Jamba hidden states to vocabulary logits.
    WARNING_TOKEN_ID=1 is reserved for danger alerts.
    """
    WARNING_TOKEN_ID: int = 1

    def __init__(self, d_jamba: int = 4096, vocab_size: int = 65536):
        super().__init__()
        self.proj = nn.Linear(d_jamba, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, S, d_jamba) -> logits: (B, S, vocab_size)"""
        return self.proj(h)
