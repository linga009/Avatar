# med_rai/training/phase3_train.py
"""
Phase 3: Joint fine-tuning of all trainable components.
Trains: QLoRA on Jamba + encoders + cross_modal + all heads.
Objective: L_total = 1.0*L_rfm + 0.5*L_text + 0.3*L_gesture
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, ConcatDataset
import os

from med_rai.model import MedRAI
from med_rai.data.jigsaws_dataset import JIGSAWSDataset
from med_rai.data.collate import medrai_collate

JIGSAWS_ROOT = os.environ.get("JIGSAWS_ROOT", "data/jigsaws")
CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/med_rai/checkpoints")


def run_phase3(phase2_ckpt: str = None):
    import torch
    model = MedRAI(lr=5e-5, lambda_rfm=1.0, lambda_text=0.5, lambda_gesture=0.3)
    if phase2_ckpt:
        state = torch.load(phase2_ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(state, strict=False)

    tasks = ["Knot_Tying", "Needle_Passing", "Suturing"]
    train_ds = ConcatDataset([
        JIGSAWSDataset(root=JIGSAWS_ROOT, task=t, split="train") for t in tasks
    ])
    val_ds = ConcatDataset([
        JIGSAWSDataset(root=JIGSAWS_ROOT, task=t, split="val") for t in tasks
    ])
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=2, collate_fn=medrai_collate)
    val_loader = DataLoader(val_ds, batch_size=4,
                            num_workers=2, collate_fn=medrai_collate)

    ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="phase3-{epoch}-{val/loss:.4f}",
        save_top_k=3, monitor="val/loss", every_n_train_steps=500,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu", devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=8,
        callbacks=[ckpt_cb, lr_monitor],
        logger=WandbLogger(project="med-rai", name="phase3")
               if os.environ.get("WANDB_API_KEY") else True,
    )
    trainer.fit(model, train_loader, val_loader)
    return ckpt_cb.best_model_path


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    run_phase3(ckpt)
