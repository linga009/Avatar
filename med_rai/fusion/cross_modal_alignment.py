import torch
import torch.nn as nn


class CrossModalAlignment(nn.Module):
    """
    Fuses 4 modality token sequences via self-attention (each attends to all others).

    Input token sequences (all projected to d_model):
        v_tokens:  (B, 1, d_model)   visual
        k_tokens:  (B, 1, d_model)   kinematics
        f_tokens:  (B, 1, d_model)   force/torque
        l_tokens:  (B, S, d_model)   language (variable length)

    Output:
        unified: (B, S+3, d_model)  concatenated, cross-attended token stream
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, v: torch.Tensor, k: torch.Tensor,
                f: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        # Concatenate all modality tokens: (B, S+3, d_model)
        tokens = torch.cat([v, k, f, l], dim=1)
        # Each token attends to all others (self + cross modal)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)
        return self.norm(tokens + attn_out)
