# halo_fep/training/tests/test_topic_bootstrap.py
"""Tests for Wikipedia topic bootstrap using a mock dataset."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from halo_fep.config import HaloFEPConfig
from halo_fep.training.topic_bootstrap import (
    TOPIC_KEYWORDS,
    _text_to_tokens,
    iter_wikipedia_token_batches,
)


def test_topic_keywords_covers_all_clusters():
    """Every cluster 0-7 must have at least one keyword."""
    cfg = HaloFEPConfig(n_tokens=32)
    assert set(TOPIC_KEYWORDS.keys()) == set(range(cfg.n_hidden))
    for cluster, kws in TOPIC_KEYWORDS.items():
        assert len(kws) >= 2, f"Cluster {cluster} has fewer than 2 keywords"


def test_text_to_tokens_shape():
    cfg = HaloFEPConfig(n_tokens=32)
    text = "The algorithm for machine learning involves equations and code."
    tokens = _text_to_tokens(text, n_tokens=cfg.n_tokens, d_model=cfg.d_model)
    assert tokens.shape == (cfg.n_tokens, cfg.d_model)
    assert tokens.dtype == np.float32


def test_text_to_tokens_short_text_zero_padded():
    tokens = _text_to_tokens("hi", n_tokens=8, d_model=16)
    assert tokens.shape == (8, 16)
    # Later slots should be zero (no text to fill them)
    assert np.allclose(tokens[2:], 0.0)


def test_iter_wikipedia_token_batches_with_mock():
    """iter_wikipedia_token_batches yields (n_tokens, d_model) arrays from all 8 clusters."""
    cfg = HaloFEPConfig(n_tokens=8, d_model=16)

    # One article per cluster — each article is >= 80 chars and matches its cluster's keywords
    fake_articles = [
        {"text": "research study investigation findings experiment analysis " * 3},   # cluster 0
        {"text": "algorithm programming software api system network computing " * 3},  # cluster 1
        {"text": "equation theorem proof mathematical calculus algebra formula " * 3}, # cluster 2
        {"text": "philosophy theory ethics consciousness epistemology reasoning " * 3}, # cluster 3
        {"text": "implementation code program function class compiler library " * 3},   # cluster 4
        {"text": "error failure problem diagnosis defect crash exception " * 3},        # cluster 5
        {"text": "history historical century ancient civilization war empire " * 3},    # cluster 6
        {"text": "future prediction forecast trend emerging innovation disrupt " * 3},  # cluster 7
    ]
    mock_ds = iter(fake_articles)

    with patch("halo_fep.training.topic_bootstrap.load_dataset", return_value=mock_ds):
        gen = iter_wikipedia_token_batches(cfg, seed=0, articles_per_cluster=1)
        batch = next(gen)

    assert batch.shape == (cfg.n_tokens, cfg.d_model)
    assert batch.dtype == np.float32
    # Verify actual content was embedded (not all zeros from fallback)
    assert not np.allclose(batch, 0.0), "batch should contain embeddings, not zero fallback"


def test_text_to_tokens_empty_text_returns_zeros():
    tokens = _text_to_tokens("", n_tokens=8, d_model=16)
    assert tokens.shape == (8, 16)
    assert np.allclose(tokens, 0.0)
