# halo/embeddings/modality_encoders.py
import math
import torch
import torch.nn as nn
import open_clip
from halo.config import HaloConfig


class TextEncoder(nn.Module):
    """Frozen CLIP text encoder + trainable linear probe.

    Conformal dimension delta_text is a learnable scalar (stored as log_delta).
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained
        )
        self.clip = model
        for p in self.clip.parameters():
            p.requires_grad_(False)

        self.probe = nn.Linear(cfg.text_embed_dim, cfg.d_model)
        self.log_delta = nn.Parameter(torch.tensor(math.log(cfg.delta_text)))

    @property
    def delta(self) -> torch.Tensor:
        return torch.exp(self.log_delta)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) integer token ids
        Returns:
            h: (B, d_model) text embeddings
        """
        with torch.no_grad():
            feat = self.clip.encode_text(token_ids)  # (B, text_embed_dim)
        return self.probe(feat.float())


class ImageEncoder(nn.Module):
    """Frozen CLIP image encoder + trainable linear probe.

    Conformal dimension delta_image is a learnable scalar (stored as log_delta).
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            cfg.clip_model, pretrained=cfg.clip_pretrained
        )
        self.clip = model
        for p in self.clip.parameters():
            p.requires_grad_(False)

        self.probe = nn.Linear(cfg.image_embed_dim, cfg.d_model)
        self.log_delta = nn.Parameter(torch.tensor(math.log(cfg.delta_image)))

    @property
    def delta(self) -> torch.Tensor:
        return torch.exp(self.log_delta)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) normalized image tensors
        Returns:
            h: (B, d_model) image embeddings
        """
        with torch.no_grad():
            feat = self.clip.encode_image(images)  # (B, image_embed_dim)
        return self.probe(feat.float())
