# halo/data/collate.py
import torch
from typing import Any


def halo_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack all tensor values in the batch dict."""
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}
