"""Episode schema for the research monitor."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class Episode:
    query: str
    order_param: float              # mean r
    mode: str                       # "explore" / "exploit"
    finding: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tokens: np.ndarray | None = None       # (n_tokens, d_model)
    query_embed: np.ndarray | None = None  # (384,) for FAISS
    free_energy_delta: float = 0.0
    audio_codes: list[int] | None = None   # VQ-VAE codebook indices (8 ints)
    vision_codes: list[int] | None = None  # VQ-VAE codebook indices (4 ints)
