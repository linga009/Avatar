# halo/tests/test_data.py
import torch
import pytest
from torch.utils.data import DataLoader
from halo.config import HaloConfig
from halo.data.synthetic_dataset import SyntheticMultimodalDataset
from halo.data.collate import halo_collate


@pytest.fixture
def cfg():
    return HaloConfig(d_model=256, text_embed_dim=768, image_embed_dim=768)


def test_dataset_length(cfg):
    ds = SyntheticMultimodalDataset(n_samples=100, cfg=cfg, seed=42)
    assert len(ds) == 100


def test_dataset_item_keys(cfg):
    ds = SyntheticMultimodalDataset(n_samples=10, cfg=cfg, seed=0)
    item = ds[0]
    assert "text_embed" in item
    assert "image_embed" in item


def test_dataset_shapes(cfg):
    ds = SyntheticMultimodalDataset(n_samples=10, cfg=cfg, seed=0)
    item = ds[0]
    assert item["text_embed"].shape == (cfg.text_embed_dim,)
    assert item["image_embed"].shape == (cfg.image_embed_dim,)


def test_cross_modal_correlation(cfg):
    """Image embed should be correlated with text embed (by construction)."""
    ds = SyntheticMultimodalDataset(n_samples=200, cfg=cfg, seed=1)
    texts  = torch.stack([ds[i]["text_embed"] for i in range(200)])
    images = torch.stack([ds[i]["image_embed"] for i in range(200)])
    # Project to 1D and check correlation > 0.1
    t1 = texts[:, 0]
    i1 = images[:, 0]
    corr = torch.corrcoef(torch.stack([t1, i1]))[0, 1].item()
    assert abs(corr) > 0.1, f"Expected cross-modal correlation, got {corr:.3f}"


def test_collate_batch_shapes(cfg):
    ds = SyntheticMultimodalDataset(n_samples=8, cfg=cfg, seed=5)
    loader = DataLoader(ds, batch_size=4, collate_fn=halo_collate)
    batch = next(iter(loader))
    assert batch["text_embed"].shape == (4, cfg.text_embed_dim)
    assert batch["image_embed"].shape == (4, cfg.image_embed_dim)
