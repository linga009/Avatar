# FineWeb-Edu Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Avatar's web search perception with FineWeb-Edu local Parquet, and add FineWeb batch training as Phase 4 of the body dream cycle.

**Architecture:** `ParquetSource` loads 3 Parquet shards at startup, builds a keyword index, and returns `list[SearchResult]` — the same interface as `web_search()`, so no downstream changes are needed. `fineweb_dream_phase()` runs 40 CLion gradient steps on random FineWeb rows as Phase 4 of the dream subprocess.

**Tech Stack:** pyarrow (already in Docker image), huggingface_hub (host only), JAX/equinox, optax CLion

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/download_fineweb.py` | Create | Download 3 Parquet shards to `data/fineweb/` on host |
| `halo3/perception/parquet_source.py` | Create | Load Parquet, keyword index, `search()` → `list[SearchResult]` |
| `tests/perception/test_parquet_source.py` | Create | Unit tests for ParquetSource |
| `halo3/perception/pipeline.py` | Modify | Auto-detect `data/fineweb/`, use ParquetSource when present |
| `halo3/training/dream_fineweb.py` | Create | Phase 4 dream: 40 CLion steps on FineWeb rows |
| `halo3/training/dream_worker.py` | Modify | Add `--fineweb-steps` arg, call `fineweb_dream_phase` |
| `halo3/main.py` | Modify | Pass `--fineweb-steps 40` to dream subprocess |

---

## Task 1: Download script

**Files:**
- Create: `scripts/download_fineweb.py`

- [ ] **Step 1: Create the download script**

```python
#!/usr/bin/env python3
"""Download 3 FineWeb-Edu Parquet shards to data/fineweb/.

Run on the HOST (not in Docker):
    python scripts/download_fineweb.py
"""
from __future__ import annotations
import os
import sys

TARGET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "fineweb"))
REPO_ID = "HuggingFaceFW/fineweb-edu"
ALLOW_PATTERNS = [
    "sample-10BT/data/train-00000-of-00096.parquet",
    "sample-10BT/data/train-00001-of-00096.parquet",
    "sample-10BT/data/train-00002-of-00096.parquet",
]


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub first:  pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Downloading 3 FineWeb-Edu shards to {TARGET_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
    )

    import glob
    files = sorted(glob.glob(os.path.join(TARGET_DIR, "**/*.parquet"), recursive=True))
    print(f"\nDone. {len(files)} shard(s):")
    for f in files:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.relpath(f, TARGET_DIR)}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
cd D:/New_Ai/.worktrees/halo3
git add scripts/download_fineweb.py
git commit -m "feat: add FineWeb-Edu download script (3 shards, ~1.2GB)"
```

---

## Task 2: ParquetSource + tests

**Files:**
- Create: `halo3/perception/parquet_source.py`
- Create: `tests/perception/test_parquet_source.py`
- Create: `tests/__init__.py`, `tests/perception/__init__.py`

- [ ] **Step 1: Write the failing tests first**

Create `tests/__init__.py` and `tests/perception/__init__.py` (empty), then:

```python
# tests/perception/test_parquet_source.py
"""Unit tests for ParquetSource."""
from __future__ import annotations
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from halo3.perception.web_fetch import SearchResult


def _make_parquet(dir_path: str) -> None:
    """Write a minimal test Parquet shard."""
    table = pa.table({
        "text": [
            "quantum computing uses quantum mechanics to process information efficiently",
            "photosynthesis converts sunlight into chemical energy in green plants",
            "neural networks learn complex patterns from large amounts of training data",
            "mitochondria is the powerhouse of the cell biology fundamentals",
            "gravitational waves detected by LIGO observatory Einstein relativity",
        ],
        "url": [
            "http://a.com", "http://b.com", "http://c.com",
            "http://d.com", "http://e.com",
        ],
        "int_score": [4, 3, 5, 4, 3],
    })
    pq.write_table(table, os.path.join(dir_path, "test.parquet"))


def test_loads_rows_and_builds_index():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        assert len(src._texts) == 5
        assert "quantum" in src._index


def test_search_returns_keyword_match():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        results = src.search("quantum computing", n=3)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert "quantum" in results[0].snippet.lower()


def test_search_sequential_fallback():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        results = src.search("zzzzzzzzz", n=2)  # no keyword match
        assert len(results) == 2
        assert src._cursor == 2


def test_filters_low_int_score():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        table = pa.table({
            "text": [
                "high quality educational content about science and learning",
                "low quality content here",
            ],
            "url": ["http://good.com", "http://bad.com"],
            "int_score": [4, 1],
        })
        pq.write_table(table, os.path.join(d, "test.parquet"))
        src = ParquetSource(d)
        assert len(src._texts) == 1
        assert "high quality" in src._texts[0]


def test_sample_texts():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        _make_parquet(d)
        src = ParquetSource(d)
        texts = src.sample_texts(3)
        assert len(texts) == 3
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)


def test_raises_if_no_parquet_files():
    from halo3.perception.parquet_source import ParquetSource
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            ParquetSource(d)
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

```bash
cd D:/New_Ai/.worktrees/halo3
python -m pytest tests/perception/test_parquet_source.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'halo3.perception.parquet_source'`

- [ ] **Step 3: Implement ParquetSource**

```python
# halo3/perception/parquet_source.py
"""ParquetSource — local FineWeb-Edu Parquet as perception source.

Replaces web_search() for per-tick organism perception.
Returns list[SearchResult] with identical interface to web_fetch.web_search().
"""
from __future__ import annotations
import glob
import logging
import os
import re
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from halo3.perception.web_fetch import SearchResult

log = logging.getLogger(__name__)

_MIN_SCORE = 3       # int_score threshold for quality filtering
_MIN_WORD_LEN = 4    # minimum word length for keyword index
_SNIPPET_LEN = 500   # chars to use for SearchResult.snippet


class ParquetSource:
    """Local FineWeb-Edu Parquet perception source.

    Loads all *.parquet files found recursively under parquet_dir at startup,
    filters by int_score >= 3, builds an inverted keyword index, and returns
    SearchResult lists on each search() call — same interface as web_search().
    """

    def __init__(self, parquet_dir: str) -> None:
        self._texts: list[str] = []
        self._urls: list[str] = []
        self._cursor: int = 0
        self._index: dict[str, list[int]] = defaultdict(list)

        files = sorted(glob.glob(
            os.path.join(parquet_dir, "**/*.parquet"), recursive=True
        ))
        if not files:
            raise FileNotFoundError(f"No Parquet files in {parquet_dir}")

        log.info(f"ParquetSource: loading {len(files)} shard(s) from {parquet_dir}")
        for path in files:
            table = pq.read_table(path, columns=["text", "url", "int_score"])
            d = table.to_pydict()
            before = len(self._texts)
            for text, url, score in zip(d["text"], d["url"], d["int_score"]):
                if (score or 0) >= _MIN_SCORE and text:
                    self._texts.append(text)
                    self._urls.append(url or "")
            log.info(f"  {os.path.basename(path)}: {len(self._texts) - before:,} rows kept")

        log.info(f"ParquetSource: {len(self._texts):,} total rows (int_score>={_MIN_SCORE})")
        self._build_index()

    def _build_index(self) -> None:
        """Build inverted keyword index: word -> [row_indices]."""
        log.info("ParquetSource: building keyword index...")
        for idx, text in enumerate(self._texts):
            for word in re.findall(r"[a-z]{4,}", text.lower()):
                self._index[word].append(idx)
            if idx > 0 and idx % 50_000 == 0:
                log.info(f"  indexed {idx:,} / {len(self._texts):,} rows")
        log.info(f"ParquetSource: index ready ({len(self._index):,} unique words)")

    def search(self, query: str, n: int = 5) -> list[SearchResult]:
        """Find rows matching query keywords. Falls back to sequential cursor."""
        words = re.findall(r"[a-z]{4,}", query.lower())

        if words:
            scores: dict[int, int] = defaultdict(int)
            for word in words:
                for idx in self._index.get(word, []):
                    scores[idx] += 1
            if scores:
                top = sorted(scores, key=lambda i: scores[i], reverse=True)[:n]
                return [self._make_result(i) for i in top]

        # Sequential cursor fallback
        total = len(self._texts)
        results = []
        for _ in range(n):
            results.append(self._make_result(self._cursor % total))
            self._cursor += 1
        return results

    def _make_result(self, idx: int) -> SearchResult:
        text = self._texts[idx]
        title = " ".join(text.split()[:8])
        snippet = text[:_SNIPPET_LEN]
        return SearchResult(title=title, snippet=snippet, url=self._urls[idx])

    def sample_texts(self, n: int) -> list[str]:
        """Return n randomly sampled texts for dream batch training."""
        indices = np.random.choice(
            len(self._texts), size=min(n, len(self._texts)), replace=False
        )
        return [self._texts[i] for i in indices]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd D:/New_Ai/.worktrees/halo3
python -m pytest tests/perception/test_parquet_source.py -v
```

Expected output:
```
PASSED tests/perception/test_parquet_source.py::test_loads_rows_and_builds_index
PASSED tests/perception/test_parquet_source.py::test_search_returns_keyword_match
PASSED tests/perception/test_parquet_source.py::test_search_sequential_fallback
PASSED tests/perception/test_parquet_source.py::test_filters_low_int_score
PASSED tests/perception/test_parquet_source.py::test_sample_texts
PASSED tests/perception/test_parquet_source.py::test_raises_if_no_parquet_files
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add halo3/perception/parquet_source.py \
        tests/__init__.py tests/perception/__init__.py \
        tests/perception/test_parquet_source.py
git commit -m "feat: add ParquetSource — FineWeb-Edu Parquet perception with keyword index"
```

---

## Task 3: Modify pipeline.py to use ParquetSource

**Files:**
- Modify: `halo3/perception/pipeline.py`

- [ ] **Step 1: Replace `__init__` and `perceive()` in pipeline.py**

The current `__init__` (lines 18–33) and `perceive()` (lines 35–44) become:

```python
# halo3/perception/pipeline.py  — full file replacement
"""Perception pipeline — orchestrates fetch → embed → token tensor.

Uses FineWeb-Edu ParquetSource when data/fineweb/ contains Parquet files.
Falls back to web search when no local data is present.
"""
from __future__ import annotations
import glob
import logging
import os

import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


class PerceptionPipeline:
    """Fetch content, embed, produce (n_tokens, d_model) tensor."""

    def __init__(self, d_model: int, n_tokens: int, vocab_size: int = 8000) -> None:
        self.n_tokens = n_tokens
        self.d_model = d_model
        self._parquet = None

        # Auto-detect FineWeb-Edu Parquet source
        parquet_dir = "data/fineweb"
        if glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True):
            try:
                from halo3.perception.parquet_source import ParquetSource
                self._parquet = ParquetSource(parquet_dir)
                log.info("Perception: using FineWeb-Edu Parquet source (web search disabled)")
            except Exception as e:
                log.warning(f"ParquetSource failed to load: {e} — falling back to web search")

        # Embedder (always needed — used by both sources and dream_fineweb)
        try:
            from halo3.perception.native_embedder import NativeEmbedder
            self.embedder = NativeEmbedder(d_model, vocab_size, n_tokens)
            if self.embedder._native_ready:
                log.info("Perception: using organism's OWN trained embedding")
            else:
                log.info("Perception: using sentence-transformers (no LM checkpoint yet)")
        except Exception:
            from halo3.perception.embedder import TextEmbedder
            self.embedder = TextEmbedder(d_model)
            log.info("Perception: using sentence-transformers fallback")

        if self._parquet is None:
            log.info("Perception: using web search (no FineWeb data found)")

    def perceive(self, query: str, max_results: int = 5) -> tuple[jnp.ndarray, list[str]]:
        """Fetch and embed content for a query."""
        if self._parquet is not None:
            results = self._parquet.search(query, n=max_results)
        else:
            from halo3.perception.web_fetch import web_search
            results = web_search(query, max_results=max_results)

        if not results:
            log.warning(f"No results for '{query}', using zero tokens")
            return jnp.zeros((self.n_tokens, self.d_model)), []

        from halo3.perception.web_fetch import results_to_texts
        texts = results_to_texts(results)
        tokens = self.embedder.texts_to_tokens(texts, self.n_tokens)
        return tokens, texts

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for FAISS indexing."""
        return self.embedder.embed_query(query)
```

- [ ] **Step 2: Verify no import errors**

```bash
cd D:/New_Ai/.worktrees/halo3
python -c "from halo3.perception.pipeline import PerceptionPipeline; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add halo3/perception/pipeline.py
git commit -m "feat: pipeline auto-detects FineWeb Parquet, falls back to web search"
```

---

## Task 4: FineWeb dream phase

**Files:**
- Create: `halo3/training/dream_fineweb.py`

- [ ] **Step 1: Create dream_fineweb.py**

```python
# halo3/training/dream_fineweb.py
"""FineWeb dream phase — Phase 4 of body dreaming.

Runs 40 CLion gradient steps on randomly sampled FineWeb-Edu rows.
Same JIT pattern as dream_replay.py: single @eqx.filter_jit defined
ONCE outside the loop; scale passed as jnp.float32() argument.
"""
from __future__ import annotations
import gc
import glob
import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

log = logging.getLogger(__name__)


def fineweb_dream_phase(
    model,
    parquet_dir: str = "data/fineweb",
    n_steps: int = 40,
    lr: float = 5e-6,
    seed: int = 99,
) -> tuple[any, dict]:
    """Train body on n_steps random FineWeb-Edu rows using CLion optimizer.

    Returns (updated_model, info_dict).
    Skips silently (returns model unchanged) if no Parquet files found.
    """
    files = glob.glob(os.path.join(parquet_dir, "**/*.parquet"), recursive=True)
    if not files:
        log.info("FineWeb dream: no Parquet files found — skipping phase 4")
        return model, {"fineweb_steps": 0, "fineweb_loss": 0.0}

    from halo3.perception.parquet_source import ParquetSource
    from halo3.perception.native_embedder import NativeEmbedder
    from halo3.loss import halo3_loss
    from halo3.training.dream_replay import scale_by_clion

    log.info(f"  FineWeb dream: loading source ({n_steps} steps, lr={lr})...")
    source = ParquetSource(parquet_dir)
    embedder = NativeEmbedder(model.cfg.d_model, n_tokens=model.cfg.n_tokens)
    texts = source.sample_texts(n_steps)

    key = jax.random.PRNGKey(seed)
    carry = model.init_carry(key)

    opt = optax.chain(
        optax.clip_by_global_norm(0.1),
        scale_by_clion(b1=0.9),
        optax.scale(-lr),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Single JIT defined ONCE outside loop — prevents XLA recompilation OOM
    @eqx.filter_jit
    def _fineweb_step(model, opt_state_in, carry, tokens, key, scale):
        loss_fn = lambda m: halo3_loss(m, carry, tokens, key)[0] * scale
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state_in,
            eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss

    def _safe_step(model, opt_state, carry, tokens, key, scale):
        try:
            new_model, new_opt_state, loss = _fineweb_step(
                model, opt_state, carry, tokens, key, jnp.float32(scale)
            )
            loss_val = float(loss)
            if loss_val != loss_val:  # NaN check
                return model, opt_state, loss
            leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
            if any(bool(jnp.any(jnp.isnan(leaf))) for leaf in leaves):
                log.warning("  FineWeb: NaN weights after step — skipping")
                return model, opt_state, loss
            return new_model, new_opt_state, loss
        except Exception as e:
            log.warning(f"  FineWeb step exception: {e}")
            return model, opt_state, 0.0

    log.info(f"  Phase 4: FineWeb batch ({n_steps} steps, scale=0.05)...")
    total_loss = 0.0
    completed = 0
    for text in texts:
        key, sk = jax.random.split(key)
        tokens = embedder.texts_to_tokens([text], model.cfg.n_tokens)
        model, opt_state, loss = _safe_step(model, opt_state, carry, tokens, sk, 0.05)
        total_loss += float(loss)
        completed += 1

    avg_loss = total_loss / max(completed, 1)
    gc.collect()
    jax.clear_caches()
    log.info(
        f"  FineWeb phase done: {completed}/{n_steps} steps | avg_loss={avg_loss:.2e}"
    )
    return model, {"fineweb_steps": completed, "fineweb_loss": avg_loss}
```

- [ ] **Step 2: Verify import**

```bash
cd D:/New_Ai/.worktrees/halo3
python -c "from halo3.training.dream_fineweb import fineweb_dream_phase; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add halo3/training/dream_fineweb.py
git commit -m "feat: add FineWeb dream phase (Phase 4 — 40 CLion steps on educational text)"
```

---

## Task 5: Wire up dream_worker.py and main.py

**Files:**
- Modify: `halo3/training/dream_worker.py` (lines 34–38, 77–93)
- Modify: `halo3/main.py` (lines 293–301)

- [ ] **Step 1: Add `--fineweb-steps` to dream_worker.py**

Add this argument after the existing `--imagine-steps` arg (after line 37):

```python
    parser.add_argument("--fineweb-steps", type=int, default=40,
                        help="FineWeb batch steps per dream (0 = disabled)")
```

- [ ] **Step 2: Call fineweb_dream_phase in dream_worker.py**

After the `save_checkpoint` call (after line 87), add:

```python
    # --- Phase 4: FineWeb batch training ---
    if args.fineweb_steps > 0:
        from halo3.training.dream_fineweb import fineweb_dream_phase
        log.info(f"  ☽ Phase 4: FineWeb batch ({args.fineweb_steps} steps)...")
        model, fw_info = fineweb_dream_phase(
            model,
            parquet_dir="data/fineweb",
            n_steps=args.fineweb_steps,
            lr=args.lr,
        )
        dream_info.update(fw_info)
        # Save again with FineWeb updates
        save_checkpoint(model, args.output)
        log.info(f"  ☽ FineWeb phase done: {fw_info}")
```

The final `dream_worker.py` should look like:

```python
"""Subprocess worker for body dreaming — runs in total memory isolation.
...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("dream_worker")

    parser = argparse.ArgumentParser(description="Body dream subprocess")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-steps", type=int, default=10)
    parser.add_argument("--recombine-steps", type=int, default=5)
    parser.add_argument("--imagine-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--fineweb-steps", type=int, default=40,
                        help="FineWeb batch steps per dream (0 = disabled)")
    args = parser.parse_args()

    import jax
    xla_cache = os.path.abspath(os.path.join("data", "xla_cache"))
    os.makedirs(xla_cache, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", xla_cache)
    log.info(f"XLA cache: {xla_cache}")

    backend = jax.default_backend()
    log.info(f"Dream worker started — JAX backend: {backend}, devices: {jax.devices()}")

    from halo3.config import Halo3Config
    from halo3.training.bootstrap import load_checkpoint, save_checkpoint

    if backend in ("gpu", "cuda"):
        cfg = Halo3Config()
    else:
        cfg = Halo3Config(
            d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
            d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
            n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
            mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
            meta_n_hidden=4, meta_n_actions=2, meta_k=3,
            max_cache=8, island_size=4,
        )

    log.info(f"Loading model from {args.checkpoint}.eqx")
    model = load_checkpoint(cfg, args.checkpoint)

    from halo3.memory.episode_store import EpisodeStore
    memory = EpisodeStore()
    episodes = memory.get_high_confidence(threshold=0.0)
    log.info(f"Loaded {len(episodes)} episodes for dreaming")

    from halo3.training.dream_replay import dream_replay_physics
    model, dream_info = dream_replay_physics(
        model, episodes,
        n_replay_steps=args.replay_steps,
        n_recombine_steps=args.recombine_steps,
        n_imagine_steps=args.imagine_steps,
        lr=args.lr,
    )

    log.info(f"Saving dreamed model to {args.output}.eqx")
    save_checkpoint(model, args.output)

    # --- Phase 4: FineWeb batch training ---
    if args.fineweb_steps > 0:
        from halo3.training.dream_fineweb import fineweb_dream_phase
        log.info(f"  ☽ Phase 4: FineWeb batch ({args.fineweb_steps} steps)...")
        model, fw_info = fineweb_dream_phase(
            model,
            parquet_dir="data/fineweb",
            n_steps=args.fineweb_steps,
            lr=args.lr,
        )
        dream_info.update(fw_info)
        save_checkpoint(model, args.output)
        log.info(f"  ☽ FineWeb phase done: {fw_info}")

    info_path = args.output + "_dream_info.json"
    with open(info_path, "w") as f:
        json.dump(dream_info, f)
    log.info(f"Dream worker done: {dream_info}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add `--fineweb-steps 40` to dream subprocess call in main.py**

Find the subprocess call (around line 293). Change:

```python
                    [sys.executable, "-m", "halo3.training.dream_worker",
                     "--checkpoint", "data/checkpoints/pre_dream",
                     "--output", "data/checkpoints/halo3",
                     "--replay-steps", "10",
                     "--recombine-steps", "5",
                     "--imagine-steps", "5"],
```

To:

```python
                    [sys.executable, "-m", "halo3.training.dream_worker",
                     "--checkpoint", "data/checkpoints/pre_dream",
                     "--output", "data/checkpoints/halo3",
                     "--replay-steps", "10",
                     "--recombine-steps", "5",
                     "--imagine-steps", "5",
                     "--fineweb-steps", "40"],
```

- [ ] **Step 4: Verify imports compile cleanly**

```bash
cd D:/New_Ai/.worktrees/halo3
python -c "
import ast, sys
for f in ['halo3/training/dream_worker.py', 'halo3/main.py']:
    ast.parse(open(f).read())
    print(f'  {f}: syntax OK')
"
```

Expected:
```
  halo3/training/dream_worker.py: syntax OK
  halo3/main.py: syntax OK
```

- [ ] **Step 5: Commit**

```bash
git add halo3/training/dream_worker.py halo3/main.py
git commit -m "feat: wire FineWeb Phase 4 into dream cycle (--fineweb-steps 40)"
```

---

## Task 6: Download data, rebuild container, restart

- [ ] **Step 1: Install huggingface_hub on host if needed**

```bash
pip install huggingface_hub
```

- [ ] **Step 2: Run download script (takes 5–15 min, ~1.2GB)**

```bash
cd D:/New_Ai/.worktrees/halo3
python scripts/download_fineweb.py
```

Expected output:
```
Downloading 3 FineWeb-Edu shards to .../data/fineweb ...
Done. 3 shard(s):
  sample-10BT/data/train-00000-of-00096.parquet  (XXX MB)
  sample-10BT/data/train-00001-of-00096.parquet  (XXX MB)
  sample-10BT/data/train-00002-of-00096.parquet  (XXX MB)
```

- [ ] **Step 3: Fix stuck query loop while rebuilding**

Delete the corrupted PFC adapter (fixes current Bug #1 immediately):

```bash
MSYS_NO_PATHCONV=1 docker exec halo3-train-1 rm -rf /app/data/pfc_adapter
```

- [ ] **Step 4: Rebuild Docker image**

```bash
cd D:/New_Ai/.worktrees/halo3
docker rm -f halo3-train-1
MSYS_NO_PATHCONV=1 docker compose build train
```

- [ ] **Step 5: Start container**

```bash
MSYS_NO_PATHCONV=1 docker compose up -d train
```

- [ ] **Step 6: Verify FineWeb source is active**

```bash
docker logs halo3-train-1 2>&1 | grep -E "ParquetSource|FineWeb|Perception"
```

Expected to see:
```
ParquetSource: loading 3 shard(s) from data/fineweb
ParquetSource: XXXXXX total rows (int_score>=3)
ParquetSource: keyword index ready (XXXXX unique words)
Perception: using FineWeb-Edu Parquet source (web search disabled)
```

- [ ] **Step 7: Verify tick speed improvement**

```bash
docker logs halo3-train-1 2>&1 | grep "Tick overrun" | tail -5
```

With FineWeb, overruns should be gone or minimal (no HTTP calls). Ticks should run at ~70s.

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "chore: FineWeb integration complete — Parquet perception + dream Phase 4"
```
