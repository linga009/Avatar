"""Pack web search results into a fixed-size (n_tokens, d_model) token array.

Layout (n_tokens=32):
  [0..3]   : 4 tokens from query embedding (tiled)
  [4..23]  : 5 results × 4 tokens (title+snippet split into 2, image fills 2)
  [24..31] : 8 remaining slots filled by image embeds or zeros
  Padding  : zeros if fewer than 5 results
"""
from __future__ import annotations

import numpy as np
from halo_fep.perception.web_fetcher import SearchResult


def pack_results(
    query_embed: np.ndarray,
    results: list[SearchResult],
    embedder,
    n_tokens: int = 32,
    d_model: int = 256,
) -> np.ndarray:
    """Returns (n_tokens, d_model) float32."""
    buf = np.zeros((n_tokens, d_model), dtype=np.float32)

    # Tokens 0-3: query tiled over 4 slots
    for i in range(min(4, n_tokens)):
        buf[i] = query_embed

    # Tokens 4-23: 5 results × 4 tokens each
    for r_idx, result in enumerate(results[:5]):
        base = 4 + r_idx * 4
        if base + 3 >= n_tokens:
            break
        title_embed   = embedder.embed_text(result.title)
        snippet_embed = embedder.embed_text(result.snippet)
        # Tokens base, base+1: text content
        buf[base]     = title_embed
        buf[base + 1] = snippet_embed
        # Tokens base+2, base+3: image (or zeros)
        img_embed     = embedder.embed_image(result.image_url)
        buf[base + 2] = img_embed
        buf[base + 3] = img_embed  # repeat for 2-token image slot

    # Tokens 24-31: extra image embeds from results
    for r_idx, result in enumerate(results[:min(4, len(results))]):
        slot = 24 + r_idx * 2
        if slot + 1 >= n_tokens:
            break
        img_embed = embedder.embed_image(result.image_url)
        buf[slot]     = img_embed
        buf[slot + 1] = img_embed

    return buf
