import torch
import torch.nn as nn
import open_clip


class SurgicalViTEncoder(nn.Module):
    """CLIP ViT-L/14 with frozen backbone and trainable linear probe."""

    def __init__(self, d_out: int = 512):
        super().__init__()
        vit, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.vit = vit.visual
        for p in self.vit.parameters():
            p.requires_grad = False
        # Detect actual output dim from the model attribute; fall back to a
        # dummy forward pass if the attribute is absent.
        vit_dim = getattr(self.vit, "output_dim", None)
        if vit_dim is None:
            import torch
            with torch.no_grad():
                vit_dim = self.vit(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.probe = nn.Linear(vit_dim, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> (B, d_out)"""
        with torch.no_grad():
            feats = self.vit(x)
        return self.probe(feats)
