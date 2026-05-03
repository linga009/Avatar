# halo/memory/page_curve_memory.py
from __future__ import annotations
from collections import deque
import torch
import torch.nn as nn
from halo.config import HaloConfig


class PageCurveMemory(nn.Module):
    """KV-cache manager based on the holographic Page curve / island formula.

    Generalized entropy per token:
        S_gen(i) = ||x_i||^2 * d_head / 4   +   H(a_i)
                   ──────────────────────────     ──────
                   area term (Bekenstein)          von Neumann entropy of attention row

    Protocol:
        - Tokens are added one at a time.
        - When active_cache exceeds max_cache, the token with minimum S_gen
          is evicted into island_buffer (FIFO, capped at island_size).
        - island_buffer tokens participate in attention as read-only context.
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        self.max_cache = cfg.max_cache
        self.island_size = cfg.island_size
        self.d_head = cfg.d_head

        # Each entry: dict with keys 'x', 'kv', 'attn'
        self.active_cache: list[dict] = []
        self.island_buffer: deque = deque(maxlen=cfg.island_size)

    def generalized_entropy(
        self,
        x_i: torch.Tensor,
        attn_i: torch.Tensor,
    ) -> torch.Tensor:
        """S_gen(i) = area_term + von_neumann_entropy.

        Args:
            x_i:   (d_boundary,) boundary position of token i
            attn_i: (N,) attention distribution of token i (sums to 1)
        Returns:
            scalar tensor
        """
        area_term = (x_i ** 2).sum() * self.d_head / 4.0
        entropy = -(attn_i * (attn_i.clamp(min=1e-8).log())).sum()
        return area_term + entropy

    def add(
        self,
        x_i: torch.Tensor,
        attn_i: torch.Tensor,
        kv_i: torch.Tensor,
    ) -> None:
        """Add a token to the cache, evicting if necessary.

        Args:
            x_i:   (d_boundary,) boundary position
            attn_i: (N,) attention row (normalized)
            kv_i:  (d_head,) KV representation to store
        """
        self.active_cache.append({"x": x_i, "kv": kv_i, "attn": attn_i})

        if len(self.active_cache) > self.max_cache:
            self._evict()

    def _evict(self) -> None:
        """Evict the token with minimum S_gen into island_buffer."""
        scores = torch.stack([
            self.generalized_entropy(entry["x"], entry["attn"])
            for entry in self.active_cache
        ])
        evict_idx = int(scores.argmin().item())
        evicted = self.active_cache.pop(evict_idx)
        self.island_buffer.append(evicted["kv"])

    def get_all_kv(self) -> torch.Tensor:
        """Return all KV tensors: active_cache + island_buffer.

        Returns:
            (N_total, d_head) stacked KV representations
        """
        active_kvs = [e["kv"] for e in self.active_cache]
        island_kvs = list(self.island_buffer)
        all_kvs = active_kvs + island_kvs
        if not all_kvs:
            return torch.zeros(0, self.d_head)
        return torch.stack(all_kvs, dim=0)

    def reset(self) -> None:
        """Clear all cached state."""
        self.active_cache.clear()
        self.island_buffer.clear()
