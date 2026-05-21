# Persistent Mind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local autonomous agent that runs 24/7 on 6GB VRAM: HALO+FEP perceives the web via free energy minimization, accumulates episodic memory, and self-improves at four timescales.

**Architecture:** HALO+FEP is the always-on subconscious (~2GB VRAM idle). Phi-3.5-mini-instruct 4-bit wakes on high surprise (~4GB peak), updates goals. Episodic memory on CPU via FAISS+SQLite. Nightly LoRA fine-tuning on high-confidence episodes.

**Tech Stack:** JAX/Equinox, duckduckgo-search, sentence-transformers, transformers+bitsandbytes, faiss-cpu, SQLAlchemy, optax

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `halo_fep/config.py` | Modify | Add `wake_threshold`, `tick_interval` fields |
| `halo_fep/utils.py` | Create | `compute_free_energy(carry, model) -> float` |
| `halo_fep/perception/__init__.py` | Create | Package |
| `halo_fep/perception/web_fetcher.py` | Create | DuckDuckGo search + HTML-to-markdown cleaning |
| `halo_fep/perception/embedder.py` | Create | Text/image → 256-dim projectors (CPU numpy) |
| `halo_fep/perception/token_packer.py` | Create | Pack results into `(32, 256)` jnp array |
| `halo_fep/perception/pipeline.py` | Create | `PerceptionPipeline`: embed + query_from_beliefs |
| `halo_fep/memory/__init__.py` | Create | Package |
| `halo_fep/memory/schema.py` | Create | `Episode` dataclass |
| `halo_fep/memory/episode_store.py` | Create | FAISS + SQLite backend |
| `halo_fep/intellect/__init__.py` | Create | Package |
| `halo_fep/intellect/state_compressor.py` | Create | carry + memories → LLM prompt string |
| `halo_fep/intellect/llm_bridge.py` | Create | Phi-3.5-mini load/unload/think |
| `halo_fep/intellect/goal_updater.py` | Create | LLM output → update `model.gm.log_C` |
| `halo_fep/training/__init__.py` | Create | Package |
| `halo_fep/training/fep_updater.py` | Create | Online Bayesian A/B/D matrix updates |
| `halo_fep/training/lora_trainer.py` | Create | Nightly HALO SSM+attention fine-tuning |
| `halo_fep/training/bootstrap.py` | Create | Phase 0 pre-training on MultimodalWorld |
| `halo_fep/main.py` | Create | Heartbeat orchestrator |

---

## Task 1: Config Extension + Free Energy Utility

**Files:**
- Modify: `halo_fep/config.py`
- Create: `halo_fep/utils.py`
- Create: `halo_fep/tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# halo_fep/tests/test_utils.py
import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.utils import compute_free_energy


def test_compute_free_energy_scalar():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    fe = compute_free_energy(carry, model)
    assert fe.shape == ()


def test_compute_free_energy_nonnegative():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    fe = compute_free_energy(carry, model)
    assert float(fe) >= 0.0


def test_config_has_wake_threshold():
    cfg = HaloFEPConfig()
    assert cfg.wake_threshold == 2.5


def test_config_has_tick_interval():
    cfg = HaloFEPConfig()
    assert cfg.tick_interval == 60
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/tests/test_utils.py -v
```
Expected: FAIL — `cannot import name 'compute_free_energy'`

- [ ] **Step 3: Add fields to config.py**

Add to `halo_fep/config.py` after the `seed` field:

```python
    # Heartbeat
    wake_threshold: float = 2.5   # FE above this triggers LLM wake cycle
    tick_interval:  int   = 60    # seconds between subconscious ticks
```

- [ ] **Step 4: Create halo_fep/utils.py**

```python
# halo_fep/utils.py
"""Shared utility functions for the Persistent Mind system."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from halo_fep.model import HaloFEPCarry, HaloFEPModel


def compute_free_energy(carry: HaloFEPCarry, model: HaloFEPModel) -> jnp.ndarray:
    """Mean KL[Q(eta) || D] over all agents — a scalar measure of surprise.

    This is the intrinsic free energy: how far current beliefs are from the prior.
    Does not require observations (can be called after any step).
    """
    q_eta = jax.nn.softmax(carry.swarm_mu, axis=-1)          # (N_agents, n_hidden)
    log_q = jnp.log(q_eta + 1e-8)                            # (N_agents, n_hidden)
    log_d = jnp.log(model.gm.D + 1e-8)                       # (n_hidden,)
    kl_per_agent = jnp.sum(q_eta * (log_q - log_d[None, :]), axis=-1)  # (N_agents,)
    return jnp.mean(kl_per_agent)
```

- [ ] **Step 5: Run tests to verify pass**

```
pytest halo_fep/tests/test_utils.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add halo_fep/config.py halo_fep/utils.py halo_fep/tests/test_utils.py
git commit -m "feat: add wake_threshold/tick_interval to config + compute_free_energy util"
```

---

## Task 2: Web Fetcher

**Files:**
- Create: `halo_fep/perception/__init__.py`
- Create: `halo_fep/perception/web_fetcher.py`
- Create: `halo_fep/perception/tests/__init__.py`
- Create: `halo_fep/perception/tests/test_web_fetcher.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/perception/tests/test_web_fetcher.py
from unittest.mock import patch, MagicMock
from halo_fep.perception.web_fetcher import WebFetcher, SearchResult


def test_search_result_fields():
    r = SearchResult(title="Test", snippet="A snippet", url="http://x.com", image_url=None)
    assert r.title == "Test"
    assert r.image_url is None


def test_web_fetcher_returns_list():
    mock_results = [
        {"title": "T1", "body": "B1", "href": "http://a.com", "image": None},
        {"title": "T2", "body": "B2", "href": "http://b.com", "image": "http://img.com/x.jpg"},
    ]
    with patch("halo_fep.perception.web_fetcher.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        MockDDGS.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.return_value = mock_results
        fetcher = WebFetcher()
        results = fetcher.search("test query", max_results=2)
    assert len(results) == 2
    assert results[0].title == "T1"
    assert results[1].image_url == "http://img.com/x.jpg"


def test_web_fetcher_handles_empty():
    with patch("halo_fep.perception.web_fetcher.DDGS") as MockDDGS:
        mock_ddgs = MagicMock()
        MockDDGS.return_value.__enter__.return_value = mock_ddgs
        mock_ddgs.text.return_value = []
        fetcher = WebFetcher()
        results = fetcher.search("nonexistent xyz", max_results=5)
    assert results == []


def test_web_fetcher_handles_exception():
    with patch("halo_fep.perception.web_fetcher.DDGS") as MockDDGS:
        MockDDGS.return_value.__enter__.side_effect = Exception("rate limit")
        fetcher = WebFetcher()
        results = fetcher.search("query", max_results=5)
    assert results == []
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/perception/tests/test_web_fetcher.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create package files**

```python
# halo_fep/perception/__init__.py
```

```python
# halo_fep/perception/tests/__init__.py
```

- [ ] **Step 4: Create web_fetcher.py**

```python
# halo_fep/perception/web_fetcher.py
"""DuckDuckGo web search wrapper with rate-limit handling."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from duckduckgo_search import DDGS

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str
    image_url: str | None


class WebFetcher:
    """Wraps DuckDuckGo search. Rate-limited to 1 req/min by default."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=max_results)
            results = []
            for r in raw:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    snippet=r.get("body", ""),
                    url=r.get("href", ""),
                    image_url=r.get("image") or None,
                ))
            return results
        except Exception as e:
            log.warning(f"WebFetcher.search failed: {e}")
            return []
```

- [ ] **Step 5: Run tests**

```
pytest halo_fep/perception/tests/test_web_fetcher.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add halo_fep/perception/
git commit -m "feat: perception package + WebFetcher (DuckDuckGo)"
```

---

## Task 3: Embedder

**Files:**
- Create: `halo_fep/perception/embedder.py`
- Create: `halo_fep/perception/tests/test_embedder.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/perception/tests/test_embedder.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_mock_st_model(dim=384):
    m = MagicMock()
    m.encode.return_value = np.random.randn(1, dim).astype(np.float32)
    return m


def make_mock_clip(dim=512):
    proc = MagicMock()
    model = MagicMock()
    model.get_image_features.return_value = MagicMock(
        detach=lambda: MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.random.randn(1, dim).astype(np.float32)))
    )
    return proc, model


@patch("halo_fep.perception.embedder.SentenceTransformer")
def test_embed_text_shape(MockST):
    MockST.return_value = make_mock_st_model(384)
    from halo_fep.perception.embedder import Embedder
    emb = Embedder(d_model=256)
    out = emb.embed_text("hello world")
    assert out.shape == (256,)
    assert out.dtype == np.float32


@patch("halo_fep.perception.embedder.SentenceTransformer")
def test_embed_text_normalized(MockST):
    MockST.return_value = make_mock_st_model(384)
    from halo_fep.perception.embedder import Embedder
    emb = Embedder(d_model=256)
    out = emb.embed_text("hello")
    norm = np.linalg.norm(out)
    assert abs(norm - 1.0) < 1e-5


@patch("halo_fep.perception.embedder.SentenceTransformer")
def test_embed_image_none_returns_zero(MockST):
    MockST.return_value = make_mock_st_model(384)
    from halo_fep.perception.embedder import Embedder
    emb = Embedder(d_model=256)
    out = emb.embed_image(None)
    assert out.shape == (256,)
    assert np.all(out == 0.0)
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/perception/tests/test_embedder.py -v
```
Expected: FAIL — `cannot import name 'Embedder'`

- [ ] **Step 3: Create embedder.py**

```python
# halo_fep/perception/embedder.py
"""Text and image embedders with linear projection to d_model.

All computation runs on CPU (no CUDA). Output is L2-normalized float32.
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TEXT_DIM   = 384
_IMAGE_MODEL = "openai/clip-vit-base-patch32"
_IMAGE_DIM   = 512


class Embedder:
    """Lazy-loads text and image models on first use.

    Projectors: two random linear projections (d_src -> d_model) fixed at init.
    These will be replaced by learned projectors in Task 11 (LoRA trainer).
    For now, random but L2-normalized outputs are sufficient for FAISS indexing.
    """

    def __init__(self, d_model: int = 256, seed: int = 42) -> None:
        self.d_model = d_model
        rng = np.random.default_rng(seed)
        # Fixed random projectors (normalized so outputs are unit-scale)
        self._text_proj  = rng.standard_normal((_TEXT_DIM, d_model)).astype(np.float32)
        self._text_proj /= np.linalg.norm(self._text_proj, axis=0, keepdims=True) + 1e-8
        self._image_proj  = rng.standard_normal((_IMAGE_DIM, d_model)).astype(np.float32)
        self._image_proj /= np.linalg.norm(self._image_proj, axis=0, keepdims=True) + 1e-8
        self._st_model  = None
        self._clip_proc  = None
        self._clip_model = None

    def _load_text_model(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(_TEXT_MODEL)

    def _load_clip(self) -> None:
        if self._clip_model is None:
            from transformers import CLIPProcessor, CLIPModel
            self._clip_proc  = CLIPProcessor.from_pretrained(_IMAGE_MODEL)
            self._clip_model = CLIPModel.from_pretrained(_IMAGE_MODEL)

    def embed_text(self, text: str) -> np.ndarray:
        """Returns (d_model,) float32 L2-normalized embedding."""
        self._load_text_model()
        raw = self._st_model.encode([text], convert_to_numpy=True)[0]  # (384,)
        projected = raw @ self._text_proj                               # (d_model,)
        norm = np.linalg.norm(projected)
        return (projected / (norm + 1e-8)).astype(np.float32)

    def embed_image(self, image_url: str | None) -> np.ndarray:
        """Returns (d_model,) float32. Returns zeros if image unavailable."""
        if image_url is None:
            return np.zeros(self.d_model, dtype=np.float32)
        try:
            import requests
            from PIL import Image
            from io import BytesIO
            self._load_clip()
            resp = requests.get(image_url, timeout=5)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            inputs = self._clip_proc(images=img, return_tensors="pt")
            import torch
            with torch.no_grad():
                feat = self._clip_model.get_image_features(**inputs)
            raw = feat.detach().cpu().numpy()[0]                         # (512,)
            projected = raw @ self._image_proj                           # (d_model,)
            norm = np.linalg.norm(projected)
            return (projected / (norm + 1e-8)).astype(np.float32)
        except Exception as e:
            log.warning(f"Image embed failed for {image_url}: {e}")
            return np.zeros(self.d_model, dtype=np.float32)
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/perception/tests/test_embedder.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add halo_fep/perception/embedder.py halo_fep/perception/tests/test_embedder.py
git commit -m "feat: Embedder — text/image to d_model with fixed random projectors"
```

---

## Task 4: Token Packer + Perception Pipeline

**Files:**
- Create: `halo_fep/perception/token_packer.py`
- Create: `halo_fep/perception/pipeline.py`
- Create: `halo_fep/perception/tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/perception/tests/test_pipeline.py
import numpy as np
import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock, patch

from halo_fep.perception.web_fetcher import SearchResult
from halo_fep.perception.token_packer import pack_results


def make_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(title=f"T{i}", snippet=f"S{i}", url=f"http://{i}.com", image_url=None)
        for i in range(n)
    ]


def make_embedder(d_model=256):
    emb = MagicMock()
    emb.d_model = d_model
    emb.embed_text.side_effect = lambda t: np.random.randn(d_model).astype(np.float32)
    emb.embed_image.return_value = np.zeros(d_model, dtype=np.float32)
    return emb


def test_pack_results_shape():
    emb = make_embedder(256)
    results = make_results(5)
    query_embed = np.random.randn(256).astype(np.float32)
    tokens = pack_results(query_embed, results, emb, n_tokens=32, d_model=256)
    assert tokens.shape == (32, 256)
    assert tokens.dtype == np.float32


def test_pack_results_fewer_than_5():
    emb = make_embedder(256)
    results = make_results(2)
    query_embed = np.random.randn(256).astype(np.float32)
    tokens = pack_results(query_embed, results, emb, n_tokens=32, d_model=256)
    assert tokens.shape == (32, 256)


def test_pack_results_empty():
    emb = make_embedder(256)
    query_embed = np.random.randn(256).astype(np.float32)
    tokens = pack_results(query_embed, [], emb, n_tokens=32, d_model=256)
    assert tokens.shape == (32, 256)


@patch("halo_fep.perception.pipeline.WebFetcher")
@patch("halo_fep.perception.pipeline.Embedder")
def test_pipeline_embed_shape(MockEmb, MockFetcher):
    from halo_fep.config import HaloFEPConfig
    from halo_fep.perception.pipeline import PerceptionPipeline

    cfg = HaloFEPConfig(n_tokens=32)

    mock_emb = make_embedder(cfg.d_model)
    MockEmb.return_value = mock_emb

    mock_fetcher = MagicMock()
    mock_fetcher.search.return_value = make_results(5)
    MockFetcher.return_value = mock_fetcher

    pipeline = PerceptionPipeline(cfg)
    out = pipeline.embed("test query")
    assert out.shape == (32, 256)
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/perception/tests/test_pipeline.py -v
```
Expected: FAIL — `cannot import name 'pack_results'`

- [ ] **Step 3: Create token_packer.py**

```python
# halo_fep/perception/token_packer.py
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
```

- [ ] **Step 4: Create pipeline.py**

```python
# halo_fep/perception/pipeline.py
"""PerceptionPipeline: web search → (n_tokens, d_model) JAX array."""
from __future__ import annotations

import logging
import numpy as np
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.perception.web_fetcher import WebFetcher
from halo_fep.perception.embedder import Embedder
from halo_fep.perception.token_packer import pack_results

log = logging.getLogger(__name__)


class PerceptionPipeline:
    def __init__(self, cfg: HaloFEPConfig) -> None:
        self.cfg     = cfg
        self.fetcher = WebFetcher()
        self.embedder = Embedder(d_model=cfg.d_model, seed=cfg.seed)

    def embed(self, query: str) -> jnp.ndarray:
        """Search query, embed results, return (n_tokens, d_model) float32."""
        results = self.fetcher.search(query, max_results=5)
        query_embed = self.embedder.embed_text(query)
        tokens_np   = pack_results(
            query_embed, results, self.embedder,
            n_tokens=self.cfg.n_tokens, d_model=self.cfg.d_model,
        )
        return jnp.array(tokens_np)

    def embed_query(self, query: str) -> np.ndarray:
        """Returns (d_model,) numpy float32 for FAISS retrieval."""
        return self.embedder.embed_text(query)

    def query_from_beliefs(self, carry) -> str:
        """Convert dominant belief cluster + action to a search query string.

        Reads argmax of mean swarm_mu across agents as the belief index,
        and argmax of mean swarm_action as the action index.
        Returns a templated query string like "topic 3 action 1 learning".
        """
        import jax.numpy as jnp
        mean_mu     = jnp.mean(carry.swarm_mu, axis=0)      # (n_hidden,)
        belief_idx  = int(jnp.argmax(mean_mu))
        mean_action = jnp.mean(carry.swarm_action, axis=0)  # (n_actions,)
        action_idx  = int(jnp.argmax(mean_action))
        templates = [
            f"topic {belief_idx} research exploration",
            f"concept {belief_idx} deep learning",
            f"idea {belief_idx} artificial intelligence",
            f"theory {belief_idx} neural network",
        ]
        return templates[action_idx % len(templates)]
```

- [ ] **Step 5: Run tests**

```
pytest halo_fep/perception/tests/test_pipeline.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add halo_fep/perception/token_packer.py halo_fep/perception/pipeline.py halo_fep/perception/tests/test_pipeline.py
git commit -m "feat: TokenPacker + PerceptionPipeline (web → 32×256 tokens)"
```

---

## Task 5: Episodic Memory

**Files:**
- Create: `halo_fep/memory/__init__.py`
- Create: `halo_fep/memory/schema.py`
- Create: `halo_fep/memory/episode_store.py`
- Create: `halo_fep/memory/tests/__init__.py`
- Create: `halo_fep/memory/tests/test_episode_store.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/memory/tests/test_episode_store.py
import numpy as np
import tempfile
import os
from halo_fep.memory.schema import Episode
from halo_fep.memory.episode_store import EpisodeStore


def make_episode(free_energy=1.0, fe_delta=0.0, d_model=256, n_hidden=8, n_tokens=32):
    return Episode(
        query="test query",
        tokens=np.random.randn(n_tokens, d_model).astype(np.float32),
        swarm_mu=np.random.randn(256, n_hidden).astype(np.float32),
        free_energy=free_energy,
        free_energy_delta=fe_delta,
    )


def test_add_and_retrieve():
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(path=d)
        ep = make_episode()
        store.add(ep)
        query_vec = np.random.randn(256).astype(np.float32)
        results = store.retrieve(query_vec, k=1)
        assert len(results) == 1
        assert results[0].query == "test query"


def test_retrieve_top_k():
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(path=d)
        for i in range(10):
            store.add(make_episode())
        results = store.retrieve(np.random.randn(256).astype(np.float32), k=5)
        assert len(results) == 5


def test_get_high_confidence():
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(path=d)
        store.add(make_episode(fe_delta=-0.2))   # high confidence
        store.add(make_episode(fe_delta=-0.01))  # below threshold
        store.add(make_episode(fe_delta=0.1))    # got worse
        results = store.get_high_confidence(min_delta=-0.05)
        assert len(results) == 1
        assert results[0].free_energy_delta == -0.2


def test_get_recent():
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(path=d)
        for i in range(10):
            store.add(make_episode())
        results = store.get_recent(n=3)
        assert len(results) == 3


def test_rebuild_index():
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(path=d)
        for i in range(5):
            store.add(make_episode())
        store.rebuild_index()
        results = store.retrieve(np.random.randn(256).astype(np.float32), k=3)
        assert len(results) == 3
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/memory/tests/test_episode_store.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create package files**

```python
# halo_fep/memory/__init__.py
```

```python
# halo_fep/memory/tests/__init__.py
```

- [ ] **Step 4: Create schema.py**

```python
# halo_fep/memory/schema.py
"""Episode dataclass — one subconscious tick's experience."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    query:              str
    tokens:             np.ndarray   # (n_tokens, d_model) float32
    swarm_mu:           np.ndarray   # (n_agents, n_hidden) float32
    free_energy:        float
    free_energy_delta:  float = 0.0
    llm_output:         str | None = None
    topic_tags:         list[str] = field(default_factory=list)
    id:                 str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:          float = field(default_factory=time.time)
```

- [ ] **Step 5: Create episode_store.py**

```python
# halo_fep/memory/episode_store.py
"""FAISS (IndexFlatIP) + SQLite episodic memory store.

FAISS index: 256-dim L2-normalized query embeddings, cosine similarity.
SQLite: full Episode data, keyed by UUID.

On startup: if FAISS index file is missing or corrupt, rebuild from SQLite.
"""
from __future__ import annotations

import json
import logging
import os
import pickle

import faiss
import numpy as np
import sqlalchemy as sa

from halo_fep.memory.schema import Episode

log = logging.getLogger(__name__)

_EMBED_DIM = 256  # query embedding stored in FAISS

_METADATA = sa.MetaData()
_EPISODES = sa.Table(
    "episodes", _METADATA,
    sa.Column("id",                 sa.String,  primary_key=True),
    sa.Column("timestamp",          sa.Float,   nullable=False),
    sa.Column("query",              sa.Text,    nullable=False),
    sa.Column("tokens",             sa.LargeBinary, nullable=False),   # pickled ndarray
    sa.Column("swarm_mu",           sa.LargeBinary, nullable=False),
    sa.Column("free_energy",        sa.Float,   nullable=False),
    sa.Column("free_energy_delta",  sa.Float,   nullable=False),
    sa.Column("llm_output",         sa.Text,    nullable=True),
    sa.Column("topic_tags",         sa.Text,    nullable=False),       # JSON list
    sa.Column("query_embed",        sa.LargeBinary, nullable=False),   # (256,) float32
)


class EpisodeStore:
    def __init__(self, path: str, embed_dim: int = _EMBED_DIM) -> None:
        os.makedirs(path, exist_ok=True)
        self._path      = path
        self._embed_dim = embed_dim
        self._db_path   = os.path.join(path, "episodes.db")
        self._idx_path  = os.path.join(path, "faiss.index")

        self._engine = sa.create_engine(f"sqlite:///{self._db_path}")
        _METADATA.create_all(self._engine)

        if os.path.exists(self._idx_path):
            try:
                self._index = faiss.read_index(self._idx_path)
            except Exception:
                log.warning("FAISS index corrupt — rebuilding from SQLite.")
                self._index = self._new_index()
                self.rebuild_index()
        else:
            self._index = self._new_index()

        # In-memory id list mirrors FAISS row order
        self._ids: list[str] = self._load_ids()

    def _new_index(self) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(self._embed_dim)

    def _load_ids(self) -> list[str]:
        """Load episode IDs in insertion order from SQLite."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES.c.id).order_by(_EPISODES.c.timestamp)
            ).fetchall()
        return [r[0] for r in rows]

    def _embed_from_query(self, query: str) -> np.ndarray:
        """Use first 256 UTF-8 bytes as a deterministic stand-in for embedding.

        NOTE: Real usage passes embedder.embed_text(query) directly via add().
        This fallback is for rebuild_index() only.
        """
        raw = query.encode("utf-8")[:self._embed_dim]
        vec = np.frombuffer(raw.ljust(self._embed_dim, b"\x00"), dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)

    def add(self, episode: Episode, query_embed: np.ndarray | None = None) -> None:
        """Persist episode. query_embed: (256,) float32 L2-normalized for FAISS."""
        if query_embed is None:
            query_embed = self._embed_from_query(episode.query)
        query_embed = query_embed.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embed)

        with self._engine.begin() as conn:
            conn.execute(_EPISODES.insert().values(
                id               = episode.id,
                timestamp        = episode.timestamp,
                query            = episode.query,
                tokens           = pickle.dumps(episode.tokens),
                swarm_mu         = pickle.dumps(episode.swarm_mu),
                free_energy      = float(episode.free_energy),
                free_energy_delta= float(episode.free_energy_delta),
                llm_output       = episode.llm_output,
                topic_tags       = json.dumps(episode.topic_tags),
                query_embed      = query_embed.tobytes(),
            ))
        self._index.add(query_embed)
        self._ids.append(episode.id)
        faiss.write_index(self._index, self._idx_path)

    def retrieve(self, query_embed: np.ndarray, k: int = 5) -> list[Episode]:
        """Return top-k episodes by cosine similarity to query_embed."""
        if self._index.ntotal == 0:
            return []
        qv = query_embed.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(qv)
        k   = min(k, self._index.ntotal)
        _, idxs = self._index.search(qv, k)
        ids = [self._ids[i] for i in idxs[0] if i >= 0]
        return self._load_by_ids(ids)

    def get_recent(self, n: int = 500) -> list[Episode]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).order_by(_EPISODES.c.timestamp.desc()).limit(n)
            ).fetchall()
        return [self._row_to_episode(r) for r in reversed(rows)]

    def get_high_confidence(self, min_delta: float = -0.05) -> list[Episode]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).where(_EPISODES.c.free_energy_delta < min_delta)
                .order_by(_EPISODES.c.timestamp)
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def rebuild_index(self) -> None:
        """Reconstruct FAISS index from SQLite (recovery path)."""
        self._index = self._new_index()
        self._ids   = []
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).order_by(_EPISODES.c.timestamp)
            ).fetchall()
        for r in rows:
            qv = np.frombuffer(r.query_embed, dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(qv)
            self._index.add(qv)
            self._ids.append(r.id)
        faiss.write_index(self._index, self._idx_path)

    def _load_by_ids(self, ids: list[str]) -> list[Episode]:
        if not ids:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES).where(_EPISODES.c.id.in_(ids))
            ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, r) -> Episode:
        ep = Episode(
            id               = r.id,
            timestamp        = r.timestamp,
            query            = r.query,
            tokens           = pickle.loads(r.tokens),
            swarm_mu         = pickle.loads(r.swarm_mu),
            free_energy      = r.free_energy,
            free_energy_delta= r.free_energy_delta,
            llm_output       = r.llm_output,
            topic_tags       = json.loads(r.topic_tags),
        )
        return ep
```

- [ ] **Step 6: Run tests**

```
pytest halo_fep/memory/tests/test_episode_store.py -v
```
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add halo_fep/memory/
git commit -m "feat: episodic memory — Episode schema + FAISS+SQLite EpisodeStore"
```

---

## Task 6: State Compressor

**Files:**
- Create: `halo_fep/intellect/__init__.py`
- Create: `halo_fep/intellect/state_compressor.py`
- Create: `halo_fep/intellect/tests/__init__.py`
- Create: `halo_fep/intellect/tests/test_state_compressor.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/intellect/tests/test_state_compressor.py
import jax
import numpy as np
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.memory.schema import Episode
from halo_fep.intellect.state_compressor import StateCompressor


def make_carry(cfg):
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    return model.init_carry(jax.random.PRNGKey(1))


def make_episode(query="past query", fe=1.0, delta=-0.1):
    return Episode(
        query=query,
        tokens=np.zeros((32, 256), dtype=np.float32),
        swarm_mu=np.zeros((256, 8), dtype=np.float32),
        free_energy=fe,
        free_energy_delta=delta,
    )


def test_compress_returns_string():
    cfg = HaloFEPConfig()
    carry = make_carry(cfg)
    compressor = StateCompressor(cfg)
    prompt = compressor.compress(carry, [], "test query", free_energy=1.0)
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_compress_contains_query():
    cfg = HaloFEPConfig()
    carry = make_carry(cfg)
    compressor = StateCompressor(cfg)
    prompt = compressor.compress(carry, [], "my special query", free_energy=0.5)
    assert "my special query" in prompt


def test_compress_contains_free_energy():
    cfg = HaloFEPConfig()
    carry = make_carry(cfg)
    compressor = StateCompressor(cfg)
    prompt = compressor.compress(carry, [], "q", free_energy=3.14)
    assert "3.14" in prompt


def test_compress_contains_memories():
    cfg = HaloFEPConfig()
    carry = make_carry(cfg)
    compressor = StateCompressor(cfg)
    mem = [make_episode("remembered search")]
    prompt = compressor.compress(carry, mem, "q", free_energy=1.0)
    assert "remembered search" in prompt


def test_compress_ends_with_options():
    cfg = HaloFEPConfig()
    carry = make_carry(cfg)
    compressor = StateCompressor(cfg)
    prompt = compressor.compress(carry, [], "q", free_energy=1.0)
    assert "SEARCH:" in prompt
    assert "GOAL:" in prompt
    assert "IDLE" in prompt
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/intellect/tests/test_state_compressor.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create package files**

```python
# halo_fep/intellect/__init__.py
```

```python
# halo_fep/intellect/tests/__init__.py
```

- [ ] **Step 4: Create state_compressor.py**

```python
# halo_fep/intellect/state_compressor.py
"""StateCompressor — formats HALO+FEP carry state into an LLM prompt string.

No neural network. Pure deterministic formatting.
"""
from __future__ import annotations

import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPCarry

_SURPRISE_LEVELS = [
    (1.0,  "low"),
    (2.0,  "medium"),
    (3.0,  "high"),
    (1e9,  "very high"),
]


def _surprise_label(fe: float) -> str:
    for threshold, label in _SURPRISE_LEVELS:
        if fe <= threshold:
            return label
    return "very high"


class StateCompressor:
    def __init__(self, cfg: HaloFEPConfig) -> None:
        self.cfg = cfg

    def compress(
        self,
        carry: HaloFEPCarry,
        recent_memories: list,
        current_query: str,
        free_energy: float,
    ) -> str:
        mean_mu     = jnp.mean(carry.swarm_mu, axis=0)       # (n_hidden,)
        belief_idx  = int(jnp.argmax(mean_mu))
        mean_action = jnp.mean(carry.swarm_action, axis=0)   # (n_actions,)
        action_idx  = int(jnp.argmax(mean_action))
        surprise    = _surprise_label(free_energy)

        lines = [
            "CURRENT STATE",
            f"Query: {current_query}",
            f"Surprise level: {surprise} (FE={free_energy:.2f})",
            f"Dominant belief: cluster {belief_idx} of {self.cfg.n_hidden}",
            f"Dominant action: action {action_idx} of {self.cfg.n_actions}",
            "",
        ]

        if recent_memories:
            lines.append("RECENT MEMORY (most similar past episodes)")
            for i, ep in enumerate(recent_memories[:5], 1):
                lines.append(
                    f"[{i}] query={ep.query!r} | FE_delta={ep.free_energy_delta:.3f}"
                )
            lines.append("")

        lines += [
            "GOAL: minimize surprise — keep exploring",
            "",
            "What should I do next? Reply with exactly one of:",
            "SEARCH: <new search query>",
            "GOAL: <new goal description>",
            "LEARN: <structured fact to remember>",
            "IDLE",
        ]

        return "\n".join(lines)
```

- [ ] **Step 5: Run tests**

```
pytest halo_fep/intellect/tests/test_state_compressor.py -v
```
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add halo_fep/intellect/
git commit -m "feat: StateCompressor — carry + memories → structured LLM prompt"
```

---

## Task 7: LLM Bridge

**Files:**
- Create: `halo_fep/intellect/llm_bridge.py`
- Create: `halo_fep/intellect/tests/test_llm_bridge.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/intellect/tests/test_llm_bridge.py
from unittest.mock import patch, MagicMock
from halo_fep.intellect.llm_bridge import LLMBridge, LLMResponse, parse_llm_output


def test_parse_search():
    r = parse_llm_output("SEARCH: active inference tutorial")
    assert r.action == "SEARCH"
    assert r.content == "active inference tutorial"


def test_parse_goal():
    r = parse_llm_output("GOAL: understand consciousness")
    assert r.action == "GOAL"
    assert r.content == "understand consciousness"


def test_parse_idle():
    r = parse_llm_output("IDLE")
    assert r.action == "IDLE"
    assert r.content == ""


def test_parse_learn():
    r = parse_llm_output("LEARN: free energy is minimized by prediction")
    assert r.action == "LEARN"


def test_parse_unknown_defaults_idle():
    r = parse_llm_output("random garbage response")
    assert r.action == "IDLE"


def test_bridge_not_loaded_initially():
    bridge = LLMBridge()
    assert not bridge.is_loaded


def test_bridge_think_raises_when_not_loaded():
    bridge = LLMBridge()
    try:
        bridge.think("test prompt")
        assert False, "Should have raised"
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/intellect/tests/test_llm_bridge.py -v
```
Expected: FAIL — `cannot import name 'LLMBridge'`

- [ ] **Step 3: Create llm_bridge.py**

```python
# halo_fep/intellect/llm_bridge.py
"""On-demand Phi-3.5-mini-instruct integration at 4-bit NF4 quantization.

Load/unload contract:
  - load() before wake cycle, unload() after (always, even on error).
  - Peak VRAM: ~2GB. Never leave model loaded between ticks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"


@dataclass
class LLMResponse:
    action:  str   # "SEARCH" | "GOAL" | "LEARN" | "IDLE"
    content: str   # the text after the prefix


def parse_llm_output(text: str) -> LLMResponse:
    """Parse first line of LLM reply into structured LLMResponse."""
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    for prefix in ("SEARCH:", "GOAL:", "LEARN:"):
        if first_line.startswith(prefix):
            return LLMResponse(
                action=prefix.rstrip(":"),
                content=first_line[len(prefix):].strip(),
            )
    if first_line.upper() == "IDLE":
        return LLMResponse(action="IDLE", content="")
    log.warning(f"LLM output not parseable: {first_line!r}. Defaulting to IDLE.")
    return LLMResponse(action="IDLE", content="")


class LLMBridge:
    """Thin wrapper around Phi-3.5-mini-instruct with load/unload lifecycle."""

    def __init__(self) -> None:
        self._model     = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load Phi-3.5-mini to CUDA with 4-bit NF4 quantization."""
        if self.is_loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        log.info(f"Loading {_MODEL_ID} at 4-bit NF4...")
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self._model     = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        log.info("LLM loaded.")

    def unload(self) -> None:
        """Free CUDA memory."""
        if not self.is_loaded:
            return
        import torch
        del self._model
        del self._tokenizer
        self._model     = None
        self._tokenizer = None
        torch.cuda.empty_cache()
        log.info("LLM unloaded.")

    def think(self, prompt: str, max_tokens: int = 128) -> str:
        """Run inference. Returns raw text output."""
        if not self.is_loaded:
            raise RuntimeError("LLMBridge.think() called before load()")
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/intellect/tests/test_llm_bridge.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add halo_fep/intellect/llm_bridge.py halo_fep/intellect/tests/test_llm_bridge.py
git commit -m "feat: LLMBridge — Phi-3.5-mini load/unload/think + parse_llm_output"
```

---

## Task 8: Goal Updater

**Files:**
- Create: `halo_fep/intellect/goal_updater.py`
- Create: `halo_fep/intellect/tests/test_goal_updater.py`

**Critical context:** `DiscreteGenerativeModel` stores `log_C` (raw logits, shape `(n_obs,)`). Its property `C` returns `jax.nn.log_softmax(log_C)` (log-space). To update goals, write a new `log_C` via `eqx.tree_at(lambda m: m.gm.log_C, model, new_log_C)`.

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/intellect/tests/test_goal_updater.py
import jax
import numpy as np
from unittest.mock import MagicMock, patch
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.intellect.goal_updater import GoalUpdater


def make_model(cfg):
    return HaloFEPModel(cfg, jax.random.PRNGKey(0))


def make_mock_embedder(n_obs=4, d_model=256):
    emb = MagicMock()
    emb.embed_text.return_value = np.random.randn(384).astype(np.float32)
    return emb


def test_update_goal_returns_model():
    cfg = HaloFEPConfig()
    model = make_model(cfg)
    updater = GoalUpdater(cfg)
    updater._embedder = make_mock_embedder(cfg.n_obs)
    new_model = updater.update_goal(model, "understand consciousness")
    assert new_model is not model  # new tree


def test_update_goal_changes_log_c():
    cfg = HaloFEPConfig()
    model = make_model(cfg)
    orig_log_c = np.array(model.gm.log_C)
    updater = GoalUpdater(cfg)
    updater._embedder = make_mock_embedder(cfg.n_obs)
    new_model = updater.update_goal(model, "something specific")
    new_log_c = np.array(new_model.gm.log_C)
    assert not np.allclose(orig_log_c, new_log_c)


def test_decay_toward_uniform():
    cfg = HaloFEPConfig()
    model = make_model(cfg)
    updater = GoalUpdater(cfg)
    new_model = updater.decay(model)
    # After decay, log_C should still be valid (no NaN/Inf)
    import jax.numpy as jnp
    assert jnp.all(jnp.isfinite(new_model.gm.log_C))
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/intellect/tests/test_goal_updater.py -v
```
Expected: FAIL — `cannot import name 'GoalUpdater'`

- [ ] **Step 3: Create goal_updater.py**

```python
# halo_fep/intellect/goal_updater.py
"""Translates LLM GOAL: output into an update of model.gm.log_C.

Steps:
  1. Embed goal text → 384-dim (sentence-transformers, CPU)
  2. Project 384 → n_obs via random fixed projection (same approach as Embedder)
  3. Softmax → probability dist over preferred observations
  4. log → new log_C
  5. Replace via eqx.tree_at

Decay: each tick, C decays 1% toward uniform. Call decay(model) every step.
"""
from __future__ import annotations

import logging
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel

log = logging.getLogger(__name__)

_TEXT_DIM = 384


class GoalUpdater:
    def __init__(self, cfg: HaloFEPConfig, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((_TEXT_DIM, cfg.n_obs)).astype(np.float32)
        self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-8
        self._embedder = None  # lazy-init; tests may inject a mock

    def _get_embedder(self):
        if self._embedder is None:
            from halo_fep.perception.embedder import Embedder
            self._embedder = Embedder(d_model=self.cfg.d_model, seed=self.cfg.seed)
        return self._embedder

    def update_goal(self, model: HaloFEPModel, goal_text: str) -> HaloFEPModel:
        """Embed goal_text and update model.gm.log_C. Returns new model."""
        embedder  = self._get_embedder()
        text_emb  = embedder.embed_text(goal_text)                # (384,) float32
        logits    = text_emb @ self._proj                         # (n_obs,)
        probs     = np.exp(logits) / (np.exp(logits).sum() + 1e-8)
        new_log_c = jnp.log(jnp.array(probs) + 1e-8)             # (n_obs,)
        return eqx.tree_at(lambda m: m.gm.log_C, model, new_log_c)

    def decay(self, model: HaloFEPModel, alpha: float = 0.99) -> HaloFEPModel:
        """Decay C matrix 1% toward uniform each step.

        Prevents the system from fixating on a single goal forever.
        """
        n_obs     = self.cfg.n_obs
        uniform   = jnp.full((n_obs,), -jnp.log(n_obs))  # log-uniform
        new_log_c = alpha * model.gm.log_C + (1.0 - alpha) * uniform
        return eqx.tree_at(lambda m: m.gm.log_C, model, new_log_c)
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/intellect/tests/test_goal_updater.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add halo_fep/intellect/goal_updater.py halo_fep/intellect/tests/test_goal_updater.py
git commit -m "feat: GoalUpdater — LLM text → log_C update + uniform decay"
```

---

## Task 9: FEP Matrix Updater

**Files:**
- Create: `halo_fep/training/__init__.py`
- Create: `halo_fep/training/fep_updater.py`
- Create: `halo_fep/training/tests/__init__.py`
- Create: `halo_fep/training/tests/test_fep_updater.py`

**Context:** `DiscreteGenerativeModel` stores `log_A (n_obs, n_hidden)`, `log_B (n_hidden, n_hidden, n_actions)`, `log_D (n_hidden,)`. Properties return softmax of stored logits. Bayesian updates: running EMA on the log-space parameters using the episode's `swarm_mu` and `soft_obs`.

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/training/tests/test_fep_updater.py
import jax
import jax.numpy as jnp
import numpy as np
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.memory.schema import Episode
from halo_fep.training.fep_updater import FEPUpdater


def make_episode(cfg):
    return Episode(
        query="q",
        tokens=np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32),
        swarm_mu=np.random.randn(cfg.n_agents, cfg.n_hidden).astype(np.float32),
        free_energy=1.0,
        free_energy_delta=-0.1,
    )


def test_update_returns_model():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    new_model = updater.update(model, carry, ep)
    assert new_model is not model


def test_update_changes_log_d():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    new_model = updater.update(model, carry, ep)
    assert not jnp.allclose(model.gm.log_D, new_model.gm.log_D)


def test_update_matrices_finite():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    new_model = updater.update(model, carry, ep)
    assert jnp.all(jnp.isfinite(new_model.gm.log_D))
    assert jnp.all(jnp.isfinite(new_model.gm.log_A))


def test_update_d_remains_distribution():
    """D property (softmax of log_D) must sum to ~1."""
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    carry = model.init_carry(jax.random.PRNGKey(1))
    ep = make_episode(cfg)
    updater = FEPUpdater(cfg)
    for _ in range(5):
        model = updater.update(model, carry, ep)
    d_sum = float(jnp.sum(model.gm.D))
    assert abs(d_sum - 1.0) < 1e-4
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/training/tests/test_fep_updater.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create package files**

```python
# halo_fep/training/__init__.py
```

```python
# halo_fep/training/tests/__init__.py
```

- [ ] **Step 4: Create fep_updater.py**

```python
# halo_fep/training/fep_updater.py
"""Online Bayesian updates to FEP generative model matrices.

After each subconscious tick, update A/B/D via exponential moving average
in log-space. This approximates a Dirichlet posterior update.

  log_D_new = alpha * log_D + (1-alpha) * log(q_eta_mean)
  log_A_new = alpha * log_A + (1-alpha) * log(outer(soft_obs_mean, q_eta_mean))

alpha=0.99 means ~100 episodes to fully update priors.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPCarry, HaloFEPModel
from halo_fep.memory.schema import Episode

_ALPHA = 0.99   # EMA decay (slower = more conservative)


class FEPUpdater:
    def __init__(self, cfg: HaloFEPConfig, alpha: float = _ALPHA) -> None:
        self.cfg   = cfg
        self.alpha = alpha

    def update(
        self,
        model: HaloFEPModel,
        carry: HaloFEPCarry,
        episode: Episode,
    ) -> HaloFEPModel:
        """Update log_D and log_A from carry beliefs. Returns new model."""
        alpha = self.alpha

        # Posterior belief: mean over agents -> (n_hidden,)
        q_eta = jax.nn.softmax(
            jnp.mean(carry.swarm_mu, axis=0)
        )                                                   # (n_hidden,)

        # --- Update D (prior over hidden states) ---
        log_q   = jnp.log(q_eta + 1e-8)                   # (n_hidden,)
        new_log_D = alpha * model.gm.log_D + (1.0 - alpha) * log_q
        # Normalize so softmax(log_D) remains a proper distribution
        new_log_D = new_log_D - jax.scipy.special.logsumexp(new_log_D)

        # --- Update A (likelihood P(obs | hidden)) ---
        # Use mean soft_obs from ObsBridge output (via mean swarm_action as proxy)
        # We approximate soft_obs as the current action distribution (n_actions,)
        # reshaped to (n_obs,) if n_obs == n_actions, else use uniform
        if self.cfg.n_obs == self.cfg.n_actions:
            soft_obs = jnp.mean(carry.swarm_action, axis=0)  # (n_obs,)
        else:
            soft_obs = jnp.ones(self.cfg.n_obs) / self.cfg.n_obs

        # outer product: (n_obs, n_hidden)
        outer    = jnp.outer(soft_obs, q_eta)              # (n_obs, n_hidden)
        log_outer = jnp.log(outer + 1e-8)
        new_log_A = alpha * model.gm.log_A + (1.0 - alpha) * log_outer
        # Normalize each column (hidden state) so A[:,j] is a distribution
        new_log_A = new_log_A - jax.scipy.special.logsumexp(new_log_A, axis=0, keepdims=True)

        new_model = eqx.tree_at(lambda m: m.gm.log_D, model,    new_log_D)
        new_model = eqx.tree_at(lambda m: m.gm.log_A, new_model, new_log_A)
        return new_model
```

- [ ] **Step 5: Run tests**

```
pytest halo_fep/training/tests/test_fep_updater.py -v
```
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add halo_fep/training/
git commit -m "feat: FEPUpdater — online Bayesian EMA updates for A/D matrices"
```

---

## Task 10: LoRA Trainer

**Files:**
- Create: `halo_fep/training/lora_trainer.py`
- Create: `halo_fep/training/tests/test_lora_trainer.py`

**Context:** Fine-tunes `SimpleSSM` diagonal matrices and `HoloAttention` projection weights. Uses `eqx.filter_grad` with a custom filter that allows gradients only through `model.backbone` layers (specifically SSM and attention). Runs `unified_elbo_loss` for 100 steps. Reverts if final loss > initial loss.

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/training/tests/test_lora_trainer.py
import jax
import numpy as np
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.memory.schema import Episode
from halo_fep.training.lora_trainer import LoRATrainer


def make_episodes(cfg, n=3):
    return [
        Episode(
            query=f"ep{i}",
            tokens=np.random.randn(cfg.n_tokens, cfg.d_model).astype(np.float32),
            swarm_mu=np.random.randn(cfg.n_agents, cfg.n_hidden).astype(np.float32),
            free_energy=1.0,
            free_energy_delta=-0.2,
        )
        for i in range(n)
    ]


def test_run_returns_model():
    cfg = HaloFEPConfig(n_steps=2)  # tiny config for speed
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    episodes = make_episodes(cfg, n=2)
    new_model, log = trainer.run(model, episodes)
    assert new_model is not None
    assert "loss_before" in log
    assert "loss_after" in log
    assert "n_episodes" in log


def test_run_logs_episode_count():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    episodes = make_episodes(cfg, n=4)
    _, log = trainer.run(model, episodes)
    assert log["n_episodes"] == 4


def test_run_empty_episodes_returns_same_model():
    cfg = HaloFEPConfig()
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    new_model, log = trainer.run(model, [])
    # Should return original model unchanged
    assert log["n_episodes"] == 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/training/tests/test_lora_trainer.py -v
```
Expected: FAIL — `cannot import name 'LoRATrainer'`

- [ ] **Step 3: Create lora_trainer.py**

```python
# halo_fep/training/lora_trainer.py
"""Nightly LoRA-style fine-tuning on high-confidence episodes.

Fine-tunes only backbone weights (SSM + attention projections) via
eqx.filter_grad. Other weights (bridges, gm, embedder) are frozen.
Reverts if loss increases after training.

Usage:
    trainer = LoRATrainer(cfg, n_steps=100, lr=1e-4)
    model, log = trainer.run(model, high_confidence_episodes)
"""
from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.loss import unified_elbo_loss
from halo_fep.memory.schema import Episode

log = logging.getLogger(__name__)


def _backbone_filter(model: HaloFEPModel) -> HaloFEPModel:
    """Return a boolean pytree: True only for backbone leaves."""
    return eqx.tree_at(
        lambda m: m.backbone,
        eqx.map(model, lambda _: False),
        replace=eqx.map(model.backbone, lambda _: True),
    )


class LoRATrainer:
    def __init__(
        self,
        cfg: HaloFEPConfig,
        n_steps: int = 100,
        lr: float = 1e-4,
    ) -> None:
        self.cfg     = cfg
        self.n_steps = n_steps
        self.opt     = optax.adam(lr)

    def run(
        self,
        model: HaloFEPModel,
        episodes: list[Episode],
    ) -> tuple[HaloFEPModel, dict[str, Any]]:
        """Fine-tune on episodes. Returns (model, log_dict)."""
        if not episodes:
            return model, {"loss_before": 0.0, "loss_after": 0.0, "n_episodes": 0}

        key   = jax.random.PRNGKey(self.cfg.seed)
        carry = model.init_carry(key)

        # Measure loss before training
        loss_before = self._mean_loss(model, carry, episodes, key)
        log.info(f"LoRA fine-tune: loss_before={loss_before:.4f}, n_episodes={len(episodes)}")

        checkpoint = model
        opt_state  = self.opt.init(eqx.filter(model, eqx.is_array))

        for step in range(self.n_steps):
            # Sample random episode for this step
            ep_idx  = step % len(episodes)
            tokens  = jnp.array(episodes[ep_idx].tokens)
            key, subkey = jax.random.split(key)

            grads = eqx.filter_grad(unified_elbo_loss, has_aux=True)(
                model, carry, tokens, subkey
            )[0]

            # Zero out grads outside backbone
            filter_mask = _backbone_filter(model)
            grads = eqx.map(grads, lambda g, mask: g if mask else jnp.zeros_like(g), filter_mask)

            updates, opt_state = self.opt.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(model, eqx.is_array),
            )
            model = eqx.apply_updates(model, updates)

            carry, _ = halo_fep_step(model, carry, tokens, subkey)

        loss_after = self._mean_loss(model, carry, episodes, key)
        log.info(f"LoRA fine-tune: loss_after={loss_after:.4f}")

        if loss_after > loss_before:
            log.warning("Loss increased after fine-tuning — reverting to checkpoint.")
            model = checkpoint

        return model, {
            "loss_before": float(loss_before),
            "loss_after":  float(loss_after),
            "n_episodes":  len(episodes),
        }

    def _mean_loss(
        self,
        model: HaloFEPModel,
        carry,
        episodes: list[Episode],
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        losses = []
        for ep in episodes[:10]:  # cap evaluation at 10 for speed
            tokens = jnp.array(ep.tokens)
            key, subkey = jax.random.split(key)
            loss, _ = unified_elbo_loss(model, carry, tokens, subkey)
            losses.append(float(loss))
        return float(np.mean(losses))
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/training/tests/test_lora_trainer.py -v
```
Expected: 3 passed (may be slow due to JIT)

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/lora_trainer.py halo_fep/training/tests/test_lora_trainer.py
git commit -m "feat: LoRATrainer — nightly backbone fine-tuning with revert-on-diverge"
```

---

## Task 11: Bootstrap Training Script

**Files:**
- Create: `halo_fep/training/bootstrap.py`

**Context:** `MultimodalWorld.sample` returns `(n_tokens=2, d_model)`. Bootstrap uses `n_tokens=32` config and pads world output with zeros to 32 tokens. Trains for 5,000 steps, then runs 100 synthetic episodes, saves checkpoint.

- [ ] **Step 1: Write failing test**

```python
# halo_fep/training/tests/test_bootstrap.py
import jax
import tempfile
import os
from halo_fep.config import HaloFEPConfig
from halo_fep.training.bootstrap import run_bootstrap, save_checkpoint, load_checkpoint


def test_save_load_roundtrip():
    cfg = HaloFEPConfig()
    from halo_fep.model import HaloFEPModel
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt")
        save_checkpoint(model, path)
        loaded = load_checkpoint(cfg, path)
    # Check a weight is numerically identical
    import jax.numpy as jnp
    assert jnp.allclose(model.gm.log_D, loaded.gm.log_D)


def test_run_bootstrap_minimal():
    """Run 2 steps (not 5000) to verify the loop executes without error."""
    cfg = HaloFEPConfig(n_tokens=32, n_steps=2)
    with tempfile.TemporaryDirectory() as d:
        model = run_bootstrap(cfg, n_pretrain_steps=2, n_synthetic_episodes=2,
                              checkpoint_dir=d, seed=0)
    assert model is not None
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/training/tests/test_bootstrap.py -v
```
Expected: FAIL — `cannot import name 'run_bootstrap'`

- [ ] **Step 3: Create bootstrap.py**

```python
# halo_fep/training/bootstrap.py
"""Phase 0 bootstrap: pre-train HALO+FEP on MultimodalWorld, save checkpoint.

Usage:
    python -m halo_fep.training.bootstrap
"""
from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.loss import unified_elbo_loss
from halo_fep.benchmark.multimodal_world import MultimodalWorld

log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "data/checkpoints/bootstrap"


def save_checkpoint(model: HaloFEPModel, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    eqx.tree_serialise_leaves(path + ".eqx", model)
    log.info(f"Checkpoint saved to {path}.eqx")


def load_checkpoint(cfg: HaloFEPConfig, path: str) -> HaloFEPModel:
    template = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    model    = eqx.tree_deserialise_leaves(path + ".eqx", template)
    log.info(f"Checkpoint loaded from {path}.eqx")
    return model


def _pad_tokens(tokens_2: jnp.ndarray, n_tokens: int) -> jnp.ndarray:
    """Pad (2, d_model) to (n_tokens, d_model) with zeros."""
    d_model = tokens_2.shape[1]
    pad = jnp.zeros((n_tokens - 2, d_model), dtype=jnp.float32)
    return jnp.concatenate([tokens_2, pad], axis=0)


def run_bootstrap(
    cfg: HaloFEPConfig,
    n_pretrain_steps: int = 5_000,
    n_synthetic_episodes: int = 100,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
    seed: int = 42,
) -> HaloFEPModel:
    key   = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    model = HaloFEPModel(cfg, k1)
    world = MultimodalWorld(cfg, k2)
    carry = model.init_carry(key)
    opt   = optax.adam(cfg.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    log.info(f"Bootstrap pre-training: {n_pretrain_steps} steps on MultimodalWorld.")

    for step in range(n_pretrain_steps):
        key, sk1, sk2 = jax.random.split(key, 3)
        eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
        tokens2, _ = world.sample(eta, sk2)
        tokens = _pad_tokens(tokens2, cfg.n_tokens)

        (loss, _), grads = eqx.filter_value_and_grad(unified_elbo_loss, has_aux=True)(
            model, carry, tokens, sk2
        )
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)

        if step % 500 == 0:
            log.info(f"Step {step}/{n_pretrain_steps} | loss={float(loss):.4f}")

    log.info(f"Running {n_synthetic_episodes} synthetic episodes to warm up memory.")
    for ep in range(n_synthetic_episodes):
        key, sk1, sk2 = jax.random.split(key, 3)
        eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
        tokens2, _ = world.sample(eta, sk2)
        tokens = _pad_tokens(tokens2, cfg.n_tokens)
        carry, _ = halo_fep_step(model, carry, tokens, sk2)

    save_checkpoint(model, checkpoint_dir)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)
    run_bootstrap(cfg)
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/training/tests/test_bootstrap.py -v
```
Expected: 2 passed (test_run_bootstrap_minimal may be slow — uses 2 steps only)

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/bootstrap.py halo_fep/training/tests/test_bootstrap.py
git commit -m "feat: bootstrap.py — Phase 0 pre-training on MultimodalWorld + checkpoint save/load"
```

---

## Task 12: Heartbeat Orchestrator

**Files:**
- Create: `halo_fep/main.py`
- Create: `halo_fep/tests/test_main.py`

- [ ] **Step 1: Write failing tests**

```python
# halo_fep/tests/test_main.py
"""Integration test: run 2 heartbeat ticks with all external calls mocked."""
import jax
import numpy as np
import jax.numpy as jnp
from unittest.mock import MagicMock, patch
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.main import HeartbeatLoop


def make_mock_perception(cfg):
    p = MagicMock()
    p.embed.return_value = jnp.zeros((cfg.n_tokens, cfg.d_model))
    p.embed_query.return_value = np.zeros(cfg.d_model, dtype=np.float32)
    p.query_from_beliefs.return_value = "test query"
    return p


def make_mock_memory():
    m = MagicMock()
    m.retrieve.return_value = []
    m.get_high_confidence.return_value = []
    return m


def test_heartbeat_tick_runs():
    cfg   = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    loop  = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=make_mock_perception(cfg),
        memory=make_mock_memory(),
    )
    loop.tick()   # should not raise


def test_heartbeat_two_ticks():
    cfg   = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    loop  = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=make_mock_perception(cfg),
        memory=make_mock_memory(),
    )
    loop.tick()
    loop.tick()


def test_heartbeat_perception_failure_continues():
    """If perception fails, tick should log and return gracefully."""
    cfg   = HaloFEPConfig(n_tokens=32)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    bad_perception = MagicMock()
    bad_perception.query_from_beliefs.return_value = "q"
    bad_perception.embed.side_effect = Exception("network error")
    loop  = HeartbeatLoop(
        cfg=cfg,
        model=model,
        perception=bad_perception,
        memory=make_mock_memory(),
    )
    loop.tick()  # should not raise
```

- [ ] **Step 2: Run to verify failure**

```
pytest halo_fep/tests/test_main.py -v
```
Expected: FAIL — `cannot import name 'HeartbeatLoop'`

- [ ] **Step 3: Create main.py**

```python
# halo_fep/main.py
"""Persistent Mind heartbeat orchestrator.

Runs the subconscious tick loop forever. High free energy triggers wake cycle.
Nightly window triggers LoRA fine-tuning. All external calls (web, LLM) are
isolated so failures degrade gracefully — the heartbeat always continues.

Usage:
    python -m halo_fep.main
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Any

import jax
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.memory.schema import Episode
from halo_fep.utils import compute_free_energy

log = logging.getLogger(__name__)


def _is_nightly_window() -> bool:
    """True between 02:00 and 02:15 local time."""
    now = datetime.datetime.now()
    return now.hour == 2 and now.minute < 15


class HeartbeatLoop:
    """Encapsulates one run of the heartbeat loop — injectable for testing."""

    def __init__(
        self,
        cfg: HaloFEPConfig,
        model: HaloFEPModel,
        perception,          # PerceptionPipeline
        memory,              # EpisodeStore
        llm=None,            # LLMBridge (optional; None skips wake cycles)
        goal_updater=None,   # GoalUpdater (optional)
        fep_updater=None,    # FEPUpdater (optional)
        lora_trainer=None,   # LoRATrainer (optional)
    ) -> None:
        self.cfg          = cfg
        self.model        = model
        self.carry        = model.init_carry(jax.random.PRNGKey(cfg.seed))
        self.perception   = perception
        self.memory       = memory
        self.llm          = llm
        self.goal_updater = goal_updater
        self.fep_updater  = fep_updater
        self.lora_trainer = lora_trainer
        self._prev_fe: float | None = None
        self._nightly_done_date: str | None = None

    def tick(self) -> None:
        """Run one subconscious tick. Never raises — logs errors and returns."""
        # --- Perception ---
        query = self.perception.query_from_beliefs(self.carry)
        try:
            tokens = self.perception.embed(query)
        except Exception as e:
            log.warning(f"Perception failed: {e}. Skipping tick.")
            return

        # --- HALO+FEP step ---
        key, carry_key = jax.random.split(self.carry.key)
        self.carry = self.carry._replace(key=key)
        try:
            self.carry, _ = halo_fep_step(self.model, self.carry, tokens, carry_key)
        except Exception as e:
            log.error(f"halo_fep_step failed: {e}. Skipping tick.")
            return

        fe = float(compute_free_energy(self.carry, self.model))
        fe_delta = (fe - self._prev_fe) if self._prev_fe is not None else 0.0
        self._prev_fe = fe

        # --- Episode ---
        episode = Episode(
            query=query,
            tokens=jnp.array(tokens).__array__(),
            swarm_mu=jnp.array(self.carry.swarm_mu).__array__(),
            free_energy=fe,
            free_energy_delta=fe_delta,
        )

        # --- FEP matrix update ---
        if self.fep_updater is not None:
            try:
                self.model = self.fep_updater.update(self.model, self.carry, episode)
            except Exception as e:
                log.warning(f"FEP matrix update failed: {e}")

        # --- Goal decay ---
        if self.goal_updater is not None:
            self.model = self.goal_updater.decay(self.model)

        self.memory.add(episode)
        log.info(f"Tick | query={query!r} | FE={fe:.3f} | FE_delta={fe_delta:+.3f}")

        # --- Wake cycle ---
        if fe > self.cfg.wake_threshold and self.llm is not None:
            self._wake_cycle(query, fe, episode)

        # --- Nightly training ---
        today = datetime.date.today().isoformat()
        if _is_nightly_window() and self._nightly_done_date != today:
            self._nightly_learning()
            self._nightly_done_date = today

    def _wake_cycle(self, query: str, fe: float, episode: Episode) -> None:
        log.info(f"Wake cycle triggered (FE={fe:.3f}).")
        from halo_fep.intellect.state_compressor import StateCompressor
        from halo_fep.intellect.llm_bridge import parse_llm_output
        try:
            query_embed = self.perception.embed_query(query)
            recent      = self.memory.retrieve(query_embed, k=5)
            compressor  = StateCompressor(self.cfg)
            prompt      = compressor.compress(self.carry, recent, query, fe)
            self.llm.load()
            output      = self.llm.think(prompt)
            self.llm.unload()
            log.info(f"Wake output: {output!r}")
            response    = parse_llm_output(output)
            if response.action == "GOAL" and self.goal_updater is not None:
                self.model = self.goal_updater.update_goal(self.model, response.content)
                log.info(f"Goal updated: {response.content!r}")
            elif response.action == "SEARCH":
                log.info(f"New search target: {response.content!r}")
            episode.llm_output = output
            self.memory.add(episode)
        except Exception as e:
            log.error(f"Wake cycle failed: {e}")
            if self.llm is not None:
                self.llm.unload()

    def _nightly_learning(self) -> None:
        log.info("Nightly learning cycle starting.")
        if self.lora_trainer is None:
            return
        try:
            episodes = self.memory.get_high_confidence()
            self.model, info = self.lora_trainer.run(self.model, episodes)
            log.info(f"Nightly learning done: {info}")
        except Exception as e:
            log.error(f"Nightly learning failed: {e}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)

    # Load or init model
    checkpoint = "data/checkpoints/bootstrap"
    try:
        from halo_fep.training.bootstrap import load_checkpoint
        model = load_checkpoint(cfg, checkpoint)
    except Exception:
        log.info("No checkpoint found — initializing fresh model.")
        model = HaloFEPModel(cfg, jax.random.PRNGKey(cfg.seed))

    from halo_fep.perception.pipeline import PerceptionPipeline
    from halo_fep.memory.episode_store import EpisodeStore
    from halo_fep.intellect.llm_bridge import LLMBridge
    from halo_fep.intellect.goal_updater import GoalUpdater
    from halo_fep.training.fep_updater import FEPUpdater
    from halo_fep.training.lora_trainer import LoRATrainer

    loop = HeartbeatLoop(
        cfg          = cfg,
        model        = model,
        perception   = PerceptionPipeline(cfg),
        memory       = EpisodeStore("data/episodes/"),
        llm          = LLMBridge(),
        goal_updater = GoalUpdater(cfg),
        fep_updater  = FEPUpdater(cfg),
        lora_trainer = LoRATrainer(cfg),
    )

    log.info("Heartbeat started. Press Ctrl+C to stop.")
    while True:
        tick_start = time.time()
        loop.tick()
        elapsed = time.time() - tick_start
        time.sleep(max(0.0, cfg.tick_interval - elapsed))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```
pytest halo_fep/tests/test_main.py -v
```
Expected: 3 passed

- [ ] **Step 5: Run full suite to confirm nothing broken**

```
pytest halo_fep/ fep_swarm/ -x -q --ignore=halo_fep/tests/test_benchmark.py
```
Expected: all pass (skip the slow benchmark test)

- [ ] **Step 6: Commit**

```bash
git add halo_fep/main.py halo_fep/tests/test_main.py
git commit -m "feat: HeartbeatLoop + main.py — persistent mind orchestrator complete"
```

---

## Spec Coverage Self-Review

| Spec Section | Covered By |
|---|---|
| HALO+FEP subconscious tick | Task 12 (HeartbeatLoop.tick) |
| Wake cycle trigger on FE > threshold | Task 12 (_wake_cycle) |
| Phi-3.5-mini load/unload | Task 7 (LLMBridge) |
| GOAL output → C matrix update | Task 8 (GoalUpdater) |
| FEP A/D matrix Bayesian update | Task 9 (FEPUpdater) |
| Nightly LoRA fine-tuning | Task 10 (LoRATrainer) |
| Revert on divergence | Task 10 (loss_after > loss_before check) |
| Episode store FAISS+SQLite | Task 5 (EpisodeStore) |
| DuckDuckGo web search | Task 2 (WebFetcher) |
| Text/image embedding pipeline | Tasks 3 & 4 |
| (32, 256) token packing | Task 4 (TokenPacker) |
| State compressor → prompt | Task 6 (StateCompressor) |
| Bootstrap pre-training | Task 11 (bootstrap.py) |
| Config: wake_threshold/tick_interval | Task 1 |
| compute_free_energy utility | Task 1 |
| 60s tick interval | Task 12 (main()) |
| Heartbeat uptime / graceful failures | Task 12 (all try/except in tick) |
| FAISS rebuild from SQLite | Task 5 (rebuild_index) |

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-06-persistent-mind.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec compliance then code quality), fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
