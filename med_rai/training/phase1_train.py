# med_rai/training/phase1_train.py
"""
Phase 1: Encoder & Cross-Modal Alignment Warmup.
Trains: SurgicalViTEncoder linear probe, SE3Encoder, FTEncoder, CrossModalAlignment.
Freezes: JambaBackbone, all output heads.
Objective: MSE reconstruction on se(3) kinematic sequences +
           InfoNCE contrastive loss between visual and kinematic tokens.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import os

from med_rai.model import MedRAI
from med_rai.data.jigsaws_dataset import JIGSAWSDataset
from med_rai.data.collate import medrai_collate

JIGSAWS_ROOT = os.environ.get("JIGSAWS_ROOT", "data/jigsaws")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/med_rai/checkpoints")


class Phase1Module(pl.LightningModule):
    """Wraps MedRAI for Phase 1 — only encoder params are trainable."""

    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = MedRAI()
        self._freeze_non_encoders()

    def _freeze_non_encoders(self):
        for name, param in self.model.named_parameters():
            if any(name.startswith(p) for p in
                   ["backbone", "rfm_head", "text_head", "gesture_head"]):
                param.requires_grad = False

    def _infonce_loss(self, v: torch.Tensor, k: torch.Tensor, temp: float = 0.07):
        """InfoNCE contrastive loss between visual and kinematic tokens."""
        v = F.normalize(v, dim=-1)
        k = F.normalize(k, dim=-1)
        logits = (v @ k.T) / temp
        labels = torch.arange(len(v), device=v.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def training_step(self, batch, batch_idx):
        v_tok = self.model.vision_enc(batch["images"])
        k_tok = self.model.se3_enc(batch["R"], batch["t"])
        contrastive = self._infonce_loss(v_tok, k_tok)
        recon = F.mse_loss(k_tok, batch["xi_traj"][:, 0, :])
        loss = contrastive + recon
        self.log("phase1/contrastive", contrastive)
        self.log("phase1/recon", recon)
        self.log("phase1/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=self.hparams.lr)
        except (ImportError, AttributeError):
            return torch.optim.AdamW(params, lr=self.hparams.lr)


def run_phase1():
    ds = JIGSAWSDataset(root=JIGSAWS_ROOT, task="Knot_Tying", split="train")
    val_ds = JIGSAWSDataset(root=JIGSAWS_ROOT, task="Knot_Tying", split="val")
    train_loader = DataLoader(ds, batch_size=16, shuffle=True,
                              num_workers=2, collate_fn=medrai_collate)
    val_loader = DataLoader(val_ds, batch_size=16,
                            num_workers=2, collate_fn=medrai_collate)
    module = Phase1Module(lr=1e-4)
    ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="phase1-{epoch}-{phase1/loss:.3f}",
        save_top_k=3, monitor="phase1/loss", every_n_train_steps=500,
    )
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu", devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb],
        logger=WandbLogger(project="med-rai", name="phase1")
               if os.environ.get("WANDB_API_KEY") else True,
    )
    trainer.fit(module, train_loader, val_loader)
    return ckpt_cb.best_model_path


if __name__ == "__main__":
    run_phase1()
