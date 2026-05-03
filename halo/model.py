# halo/model.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from halo.config import HaloConfig
from halo.embeddings.holo_embedding import HoloEmbedding
from halo.embeddings.modality_encoders import TextEncoder, ImageEncoder
from halo.backbone.halo_backbone import HALOBackbone
from halo.flow.ads_kg_prior import AdSKGPrior
from halo.memory.page_curve_memory import PageCurveMemory
from halo.heads.text_head import TextHead
from halo.heads.image_head import ImageHead
from halo.loss import HALOLoss


class HALOModel(pl.LightningModule):
    """Holographic AdS-Learned Omnimodal model.

    Training mode: flow matching on (text_embed, image_embed) pairs.
    The backbone takes concatenated text + image tokens and predicts
    the flow vector field v_pred. Target is v_KG (AdS prior) + residual.
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.text_encoder  = TextEncoder(cfg)
        self.image_encoder = ImageEncoder(cfg)
        self.holo_embed    = HoloEmbedding(cfg)
        self.backbone      = HALOBackbone(cfg)
        self.ads_kg_prior  = AdSKGPrior(cfg)
        self.page_memory   = PageCurveMemory(cfg)
        self.text_head     = TextHead(cfg)
        self.image_head    = ImageHead(cfg)
        self.loss_fn       = HALOLoss(cfg)

        # Residual correction network: maps (x_t, t) -> correction in d_model
        self.residual_net = nn.Sequential(
            nn.Linear(cfg.d_model + 1, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward_embeddings(
        self,
        text_h: torch.Tensor,
        image_h: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Core forward pass operating on pre-encoded embeddings.

        Args:
            text_h:  (B, N_text, d_model)
            image_h: (B, N_image, d_model)
        Returns:
            dict with v_pred, v_target, attn_weights, evict_scores
        """
        # Concatenate modalities
        h = torch.cat([text_h, image_h], dim=1)  # (B, N, d_model)
        B, N, _ = h.shape

        # Poincaré lifting
        x, z = self.holo_embed(h)  # (B, N, d_b), (B, N, 1)

        # Sample flow time t ~ U(0, 1)
        t = torch.rand(B, 1, 1, device=h.device)

        # AdS-KG prior: use x (boundary coords) as the "data" space
        x_noise = torch.randn_like(x)
        x_t, v_kg = self.ads_kg_prior.interpolate(x_noise, x, t)

        # Backbone — operates in d_model space
        # Interpolate h as well for the backbone input
        h_noise = torch.randn_like(h)
        h_t = (1 - t) * h_noise + t * h

        h_out = self.backbone(h_t, x_t, z)  # (B, N, d_model)

        # Residual correction: append t to each token
        t_expanded = t.expand(B, N, 1)
        h_with_t = torch.cat([h_out, t_expanded], dim=-1)  # (B, N, d_model+1)
        v_residual = self.residual_net(h_with_t)            # (B, N, d_model)

        # Target vector field: straight-line OT (h - h_noise)
        v_target = h - h_noise  # (B, N, d_model)

        # Predicted vector field: KG prior projected to d_model + residual
        # Project v_kg (d_boundary) back to d_model via holo_embed x_proj weight
        v_kg_dm = v_kg @ self.holo_embed.x_proj.weight  # (B, N, d_model)
        v_pred = v_kg_dm + v_residual                    # (B, N, d_model)

        # Attention weights proxy for Bekenstein loss
        with torch.no_grad():
            attn_weights = torch.ones(B, N, N, device=h.device) / N

        # Eviction scores: softmax of negative S_gen proxy
        x_flat = x.reshape(B * N, -1)          # (BN, d_b)
        area = (x_flat ** 2).sum(dim=-1) * self.cfg.d_head / 4.0
        evict_scores = torch.softmax(-area.reshape(B, N), dim=-1)

        return {
            "v_pred":        v_pred,
            "v_target":      v_target,
            "attn_weights":  attn_weights,
            "evict_scores":  evict_scores,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Project raw embeddings (embed_dim) -> d_model via encoder probes
        text_h  = self.text_encoder.probe(batch["text_embed"]).unsqueeze(1)   # (B, 1, d_model)
        image_h = self.image_encoder.probe(batch["image_embed"]).unsqueeze(1) # (B, 1, d_model)
        out = self.forward_embeddings(text_h, image_h)
        total, parts = self.loss_fn(
            out["v_pred"], out["v_target"],
            out["attn_weights"], out["evict_scores"],
        )
        self.log("train/loss",        total,          prog_bar=True)
        self.log("train/loss_fm",     parts["fm"])
        self.log("train/loss_bek",    parts["bek"])
        self.log("train/loss_thermo", parts["thermo"])
        self.log("train/loss_page",   parts["page"])
        return total

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    @torch.no_grad()
    def generate(
        self,
        text_h: torch.Tensor,
        n_image_tokens: int = 1,
    ) -> torch.Tensor:
        """Generate image embeddings from text embeddings via 4-step Euler ODE.

        Args:
            text_h: (B, N_text, d_model) text embeddings
            n_image_tokens: number of image tokens to generate
        Returns:
            image_embed: (B, n_image_tokens, image_embed_dim)
        """
        B = text_h.shape[0]
        # Start from noise in image embedding space
        x = torch.randn(B, n_image_tokens, self.cfg.d_model, device=text_h.device)

        dt = 1.0 / self.cfg.flow_steps
        for step in range(self.cfg.flow_steps):
            t_val = step * dt
            t = torch.full((B, 1, 1), t_val, device=x.device)

            combined = torch.cat([text_h, x], dim=1)
            x_bc, z_bc = self.holo_embed(combined)

            x_noise = torch.randn_like(x_bc[:, text_h.shape[1]:])
            v_kg = self.ads_kg_prior(x_noise, x_bc[:, text_h.shape[1]:], t)

            # Backbone step
            h_out = self.backbone(combined, x_bc, z_bc)
            image_out = h_out[:, text_h.shape[1]:]

            t_exp = t.expand(B, n_image_tokens, 1)
            res = self.residual_net(
                torch.cat([image_out, t_exp], dim=-1)
            )

            v_kg_dm = v_kg @ self.holo_embed.x_proj.weight
            v_total = v_kg_dm + res

            x = x + dt * v_total  # Euler step

        return self.image_head(x)  # (B, n_image_tokens, image_embed_dim)
