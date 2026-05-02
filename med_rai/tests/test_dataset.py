import torch
import os
import pytest
from med_rai.data.jigsaws_dataset import JIGSAWSDataset
from med_rai.data.collate import medrai_collate
from med_rai.data.synthetic_se3 import SyntheticSE3Dataset

JIGSAWS_ROOT = os.environ.get("JIGSAWS_ROOT", "data/jigsaws")

@pytest.mark.skipif(not os.path.exists(JIGSAWS_ROOT),
                    reason="JIGSAWS data not available")
def test_dataset_len():
    ds = JIGSAWSDataset(root=JIGSAWS_ROOT, task="Knot_Tying", split="train")
    assert len(ds) > 0

@pytest.mark.skipif(not os.path.exists(JIGSAWS_ROOT),
                    reason="JIGSAWS data not available")
def test_item_keys():
    ds = JIGSAWSDataset(root=JIGSAWS_ROOT, task="Knot_Tying", split="train")
    item = ds[0]
    for key in ("images", "R", "t", "ft", "token_ids", "xi_traj", "gesture_labels", "text_targets"):
        assert key in item, f"Missing key: {key}"

def test_synthetic_collate():
    """Collate pads variable-length token sequences correctly."""
    import numpy as np
    M1, M2 = torch.randn(3, 3), torch.randn(3, 3)
    R1 = torch.linalg.qr(M1)[0]
    R2 = torch.linalg.qr(M2)[0]
    samples = [
        {"images": torch.randn(3, 224, 224), "R": R1,
         "t": torch.randn(3), "ft": torch.randn(6),
         "token_ids": torch.randint(0, 100, (8,)),
         "xi_traj": torch.randn(10, 6),
         "gesture_labels": torch.tensor(0),
         "text_targets": torch.randint(0, 100, (11,))},
        {"images": torch.randn(3, 224, 224), "R": R2,
         "t": torch.randn(3), "ft": torch.randn(6),
         "token_ids": torch.randint(0, 100, (12,)),
         "xi_traj": torch.randn(10, 6),
         "gesture_labels": torch.tensor(3),
         "text_targets": torch.randint(0, 100, (15,))},
    ]
    batch = medrai_collate(samples)
    assert batch["token_ids"].shape[0] == 2
    assert batch["token_ids"].shape[1] == 12  # padded to max


def test_synthetic_dataset_len():
    ds = SyntheticSE3Dataset(n_samples=100)
    assert len(ds) == 100

def test_synthetic_item_shape():
    ds = SyntheticSE3Dataset(n_samples=10)
    item = ds[0]
    assert item["xi_0"].shape == (6,)
    assert item["xi_1"].shape == (6,)
    assert item["xi_traj"].shape == (10, 6)

def test_synthetic_geodesic_linearity():
    """Interpolated trajectory must be a straight line in se(3)."""
    ds = SyntheticSE3Dataset(n_samples=10)
    item = ds[0]
    xi_0, xi_1 = item["xi_0"], item["xi_1"]
    xi_traj = item["xi_traj"]
    for step in range(10):
        t = (step + 1) / 10
        expected = (1 - t) * xi_0 + t * xi_1
        assert torch.allclose(xi_traj[step], expected, atol=1e-5)
