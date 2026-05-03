# halo/baseline.py
"""
Euclidean flow matching baseline — same capacity, flat dot-product attention.

Usage:
    python halo/baseline.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from halo.config import HaloConfig
from halo.data.synthetic_dataset import SyntheticMultimodalDataset
from halo.data.collate import halo_collate


class EuclideanFlowModel(nn.Module):
    """Same parameter count as HALO but uses dot-product attention and no AdS prior."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        self.out    = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        h = self.encoder(h)
        q = self.attn_q(h)
        k = self.attn_k(h)
        v = self.attn_v(h)
        scale = D ** 0.5
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / scale, dim=-1)
        h = h + torch.bmm(attn, v)
        return self.out(h)


def train_baseline(
    d_model: int = 128, d_embed: int = 64, max_epochs: int = 20
) -> list[float]:
    cfg = HaloConfig(d_model=d_model, text_embed_dim=d_embed, image_embed_dim=d_embed)
    ds  = SyntheticMultimodalDataset(n_samples=1000, cfg=cfg, seed=42)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=halo_collate)

    model = EuclideanFlowModel(d_model=d_embed)
    opt   = torch.optim.Adam(model.parameters(), lr=3e-4)
    history = []

    for epoch in range(max_epochs):
        epoch_losses = []
        for batch in loader:
            text_h  = batch["text_embed"]   # (B, d_embed)
            image_h = batch["image_embed"]  # (B, d_embed)
            h = torch.stack([text_h, image_h], dim=1)  # (B, 2, d_embed)

            t   = torch.rand(h.shape[0], 1, 1)
            x0  = torch.randn_like(h)
            x_t = (1 - t) * x0 + t * h
            v_target = h - x0

            v_pred = model(x_t, t)
            loss   = F.mse_loss(v_pred, v_target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())

        avg = sum(epoch_losses) / len(epoch_losses)
        history.append(avg)
        print(f"Baseline epoch {epoch+1}/{max_epochs}: loss={avg:.6f}")

    return history


if __name__ == "__main__":
    print("Training Euclidean baseline...")
    history = train_baseline(max_epochs=20)
    print(f"Final baseline loss: {history[-1]:.6f}")
