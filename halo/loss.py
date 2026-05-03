# halo/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from halo.config import HaloConfig


class HALOLoss(nn.Module):
    """Combined loss for HALO training.

    L_total = L_FM + lambda_bek * L_Bek + lambda_thermo * L_thermo + lambda_page * L_page

    L_FM    — flow matching MSE between predicted and target vector fields
    L_Bek   — Bekenstein bound: penalise attention entropy > alpha * N * d_head
    L_thermo— entropy production lower bound (effective learning signal)
    L_page  — KL between model eviction distribution and S_gen-derived target
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.eps_prod_min = 0.01  # minimum entropy production for effective learning

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        attn_weights: torch.Tensor,
        evict_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            v_pred:       (B, N, d_model) predicted vector field
            v_target:     (B, N, d_model) target vector field
            attn_weights: (B, N, N) attention distribution (rows sum to 1)
            evict_scores: (B, N) normalised eviction scores from page curve
        Returns:
            total: scalar loss
            parts: dict of individual loss components
        """
        cfg = self.cfg

        # --- Flow Matching loss ---
        l_fm = F.mse_loss(v_pred, v_target)

        # --- Bekenstein regulariser ---
        # Attention entropy H(A) = -sum(a_ij * log(a_ij)) per row
        attn_entropy = -(attn_weights * (attn_weights.clamp(min=1e-8).log())).sum(dim=-1)
        # (B, N)
        B, N, _ = attn_weights.shape
        bound = cfg.bekenstein_alpha * N * cfg.d_head
        l_bek = torch.clamp(attn_entropy - bound, min=0.0).mean()

        # --- Thermodynamic entropy production lower bound ---
        # Entropy production proxy: mean squared norm of predicted flow
        eps_prod = (v_pred ** 2).sum(dim=-1).mean() / 2.0
        l_thermo = torch.clamp(self.eps_prod_min - eps_prod, min=0.0)

        # --- Page curve alignment ---
        # evict_scores is the model's implicit eviction preference;
        # uniform target = equal attention to all tokens (max entropy eviction)
        uniform = torch.ones_like(evict_scores) / evict_scores.shape[-1]
        l_page = F.kl_div(
            evict_scores.clamp(min=1e-8).log(),
            uniform,
            reduction="batchmean",
        )

        total = (
            l_fm
            + cfg.lambda_bek    * l_bek
            + cfg.lambda_thermo * l_thermo
            + cfg.lambda_page   * l_page
        )

        parts = {"fm": l_fm, "bek": l_bek, "thermo": l_thermo, "page": l_page}
        return total, parts
