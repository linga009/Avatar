# halo/train.py
"""
Train HALO on the synthetic multimodal dataset.

Usage:
    python halo/train.py
"""
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from halo.config import HaloConfig
from halo.data.synthetic_dataset import SyntheticMultimodalDataset
from halo.data.collate import halo_collate
from halo.model import HALOModel


def train(cfg: HaloConfig | None = None, max_epochs: int = 20) -> list[float]:
    if cfg is None:
        cfg = HaloConfig(
            d_model=128, d_boundary=32, n_heads=2, n_layers=4,
            d_state=8, d_ff=256, text_embed_dim=64, image_embed_dim=64,
            vocab_size=100, max_cache=32, island_size=8, lr=3e-4,
        )

    ds_train = SyntheticMultimodalDataset(n_samples=1000, cfg=cfg, seed=42)
    ds_val   = SyntheticMultimodalDataset(n_samples=200,  cfg=cfg, seed=99)

    loader_train = DataLoader(ds_train, batch_size=32, shuffle=True,
                              collate_fn=halo_collate)
    loader_val   = DataLoader(ds_val,   batch_size=32, shuffle=False,
                              collate_fn=halo_collate)

    with patch("halo.embeddings.modality_encoders.open_clip") as mock_clip:
        mock_model = MagicMock()
        mock_model.encode_text.side_effect  = lambda x: torch.randn(x.shape[0], cfg.text_embed_dim)
        mock_model.encode_image.side_effect = lambda x: torch.randn(x.shape[0], cfg.image_embed_dim)
        mock_clip.create_model_and_transforms.return_value = (
            mock_model, MagicMock(), MagicMock()
        )
        mock_clip.get_tokenizer.return_value = MagicMock()
        model = HALOModel(cfg)

    loss_history = []

    class LossCallback(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            val = trainer.callback_metrics.get("train/loss")
            if val is not None:
                loss_history.append(float(val))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        callbacks=[LossCallback()],
    )
    trainer.fit(model, loader_train, loader_val)
    return loss_history


if __name__ == "__main__":
    print("Training HALO...")
    history = train(max_epochs=20)
    print(f"Final loss: {history[-1]:.6f}")
    print(f"Loss curve: {[f'{l:.4f}' for l in history]}")
