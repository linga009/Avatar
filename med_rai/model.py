import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any

from med_rai.encoders.vision_encoder import SurgicalViTEncoder
from med_rai.encoders.se3_encoder import SE3Encoder
from med_rai.encoders.ft_encoder import FTEncoder
from med_rai.encoders.language_embedder import LanguageEmbedder
from med_rai.fusion.cross_modal_alignment import CrossModalAlignment
from med_rai.backbone.jamba_backbone import JambaBackbone
from med_rai.heads.rfm_policy_head import RFMPolicyHead
from med_rai.heads.text_head import TextHead
from med_rai.heads.gesture_head import GestureHead
from med_rai.loss import MedRAILoss
from med_rai.utils.se3_utils import clamp_to_workspace, se3_exp

WORKSPACE = {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (0.03, 0.25)}
D_HIDDEN = 512
N_STEPS_ODE = 4


class MedRAI(pl.LightningModule):
    def __init__(self, lr: float = 1e-4,
                 lambda_rfm: float = 1.0,
                 lambda_text: float = 0.5,
                 lambda_gesture: float = 0.3):
        super().__init__()
        self.save_hyperparameters()

        self.vision_enc   = SurgicalViTEncoder(d_out=D_HIDDEN)
        self.se3_enc      = SE3Encoder(d_out=D_HIDDEN)
        self.ft_enc       = FTEncoder(d_out=D_HIDDEN)
        self.lang_emb     = LanguageEmbedder(d_out=D_HIDDEN)
        self.cross_modal  = CrossModalAlignment(d_model=D_HIDDEN, n_heads=8)
        self.backbone     = JambaBackbone()
        self.rfm_head     = RFMPolicyHead(d_jamba=4096, d_hidden=D_HIDDEN)
        self.text_head    = TextHead(d_jamba=4096)
        self.gesture_head = GestureHead(d_jamba=4096)
        self.loss_fn      = MedRAILoss(lambda_rfm, lambda_text, lambda_gesture)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Encode each modality -> (B, 1, D_HIDDEN) or (B, S, D_HIDDEN)
        v_tok = self.vision_enc(batch["images"]).unsqueeze(1)       # (B, 1, D_HIDDEN)
        k_tok = self.se3_enc(batch["R"], batch["t"]).unsqueeze(1)   # (B, 1, D_HIDDEN)
        f_tok = self.ft_enc(batch["ft"]).unsqueeze(1)               # (B, 1, D_HIDDEN)
        l_tok = self.lang_emb(batch["token_ids"])                   # (B, S, D_HIDDEN)

        # Cross-modal fusion -> (B, S+3, D_HIDDEN)
        fused = self.cross_modal(v_tok, k_tok, f_tok, l_tok)

        # Project fused tokens to integer IDs for Jamba
        token_ids = fused.argmax(dim=-1).clamp(0, 999)              # (B, S+3)
        h = self.backbone(token_ids)                                 # (B, S+3, 4096)

        # ODE inference: 4-step Euler from xi_0 ~ N(0, I)
        B_size = h.shape[0]
        xi_0 = torch.randn(B_size, self.rfm_head.traj_horizon, 6, device=h.device)
        v_theta = self._ode_solve(xi_0, h)                          # (B, H, 6)

        text_logits    = self.text_head(h)                          # (B, S+3, vocab_size)
        gesture_logits = self.gesture_head(h)                       # (B, n_gestures)

        # Danger override: zero trajectory and force RETRACT gesture
        if self._check_danger(text_logits):
            return {
                "v_theta":        torch.zeros_like(v_theta),
                "text_logits":    text_logits,
                "gesture_logits": gesture_logits,
                "forced_gesture": "RETRACT",
            }

        return {
            "v_theta":        v_theta,
            "text_logits":    text_logits,
            "gesture_logits": gesture_logits,
        }

    def _ode_solve(self, xi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """4-step Euler ODE integration in se(3)."""
        dt = 1.0 / N_STEPS_ODE
        for step in range(N_STEPS_ODE):
            t = torch.full((xi.shape[0],), step * dt, device=xi.device)
            v = self.rfm_head(xi, t, h)
            xi = xi + dt * v
        # Clamp translation component to workspace after integration
        _, t_out = se3_exp(xi.reshape(-1, 6))
        t_out = clamp_to_workspace(t_out, WORKSPACE)
        # Write clamped translation back into xi (last 3 dims)
        B_size, H, _ = xi.shape
        xi = xi.clone()
        xi.reshape(B_size * H, 6)[:, 3:] = t_out
        return xi

    def _check_danger(self, text_logits: torch.Tensor) -> bool:
        """Returns True if WARNING token is the top prediction at any position."""
        preds = text_logits.argmax(dim=-1)          # (B, S)
        return bool((preds == TextHead.WARNING_TOKEN_ID).any().item())

    def training_step(self, batch, batch_idx):
        out = self(batch)
        total, components = self.loss_fn(
            out["v_theta"], batch["xi_traj"],
            out["text_logits"], batch["text_targets"],
            out["gesture_logits"], batch["gesture_labels"],
        )
        self.log("train/loss", total, prog_bar=True)
        for k, v in components.items():
            self.log(f"train/loss_{k}", v)
        return total

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        total, components = self.loss_fn(
            out["v_theta"], batch["xi_traj"],
            out["text_logits"], batch["text_targets"],
            out["gesture_logits"], batch["gesture_labels"],
        )
        self.log("val/loss", total, prog_bar=True)
        for k, v in components.items():
            self.log(f"val/loss_{k}", v)

    def configure_optimizers(self):
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(self.parameters(), lr=self.hparams.lr)
        except (ImportError, AttributeError):
            # Fallback for non-CUDA environments (Windows dev, testing)
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
