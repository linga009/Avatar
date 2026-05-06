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
