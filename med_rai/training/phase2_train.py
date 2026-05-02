# med_rai/training/phase2_train.py
"""
Phase 2: RFM Policy Head Warmup on synthetic SE(3) geodesic trajectories.
Trains: RFMPolicyHead only.
Freezes: Everything else.
Objective: Flow Matching MSE loss in se(3).
"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import os

from med_rai.model import MedRAI
from med_rai.data.synthetic_se3 import SyntheticSE3Dataset

CKPT_DIR = os.environ.get("CKPT_DIR", "/content/drive/MyDrive/med_rai/checkpoints")
D_JAMBA = 4096
TRAJ_HORIZON = 10


class Phase2Module(pl.LightningModule):
    def __init__(self, phase1_ckpt: str = None, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = MedRAI()
        if phase1_ckpt:
            state = torch.load(phase1_ckpt, map_location="cpu")["state_dict"]
            self.model.load_state_dict(state, strict=False)
        self._freeze_all_but_rfm()

    def _freeze_all_but_rfm(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("rfm_head"):
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        B = batch["xi_0"].shape[0]
        xi_0 = batch["xi_0"].unsqueeze(1).expand(-1, TRAJ_HORIZON, -1)
        xi_1 = batch["xi_1"].unsqueeze(1).expand(-1, TRAJ_HORIZON, -1)

        t_val = torch.rand(B, device=xi_0.device)
        t_exp = t_val.view(B, 1, 1).expand_as(xi_0)
        xi_t = (1 - t_exp) * xi_0 + t_exp * xi_1
        u_t = xi_1 - xi_0

        # Use zero hidden states (backbone frozen) during warmup
        h = torch.zeros(B, 1, D_JAMBA, device=xi_0.device)
        v_pred = self.model.rfm_head(xi_t, t_val, h)
        loss = ((v_pred - u_t) ** 2).mean()
        self.log("phase2/rfm_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=self.hparams.lr)
        except (ImportError, AttributeError):
            return torch.optim.AdamW(params, lr=self.hparams.lr)


def run_phase2(phase1_ckpt: str = None):
    ds = SyntheticSE3Dataset(n_samples=50000)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
    module = Phase2Module(phase1_ckpt=phase1_ckpt)
    ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="phase2-{epoch}-{phase2/rfm_loss:.4f}",
        save_top_k=3, monitor="phase2/rfm_loss", every_n_train_steps=500,
    )
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu", devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb],
    )
    trainer.fit(module, loader)
    return ckpt_cb.best_model_path


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    run_phase2(ckpt)
