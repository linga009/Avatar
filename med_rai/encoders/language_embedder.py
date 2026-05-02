import torch
import torch.nn as nn


class LanguageEmbedder(nn.Module):
    """
    Projects token IDs into d_out-dimensional embeddings.
    Embedding table is frozen; only the linear projection trains.
    """

    def __init__(self, vocab_size: int = 65536, d_jamba: int = 4096, d_out: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_jamba)
        for p in self.embed.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(d_jamba, d_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, seq_len) -> (B, seq_len, d_out)"""
        emb = self.embed(tokens)
        return self.proj(emb)
