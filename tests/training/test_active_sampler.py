"""Tests for ActiveSampler — FE-guided curriculum selection."""
from __future__ import annotations
import pytest


def test_zone_of_proximal_development():
    """select_texts_by_fe filters to medium-FE zone."""
    from halo3.training.active_sampler import select_texts_by_fe

    texts = [f"text_{i}" for i in range(20)]
    fe_scores = list(range(20))  # 0, 1, 2, ..., 19

    result = select_texts_by_fe(texts, fe_scores, n_select=6)

    assert len(result) == 6
    for text in result:
        idx = int(text.split("_")[1])
        assert 4 <= idx <= 15, f"text_{idx} outside zone of proximal development"


def test_zone_handles_small_input():
    """select_texts_by_fe handles fewer candidates than n_select."""
    from halo3.training.active_sampler import select_texts_by_fe

    texts = ["a", "b", "c"]
    fe_scores = [1.0, 2.0, 3.0]
    result = select_texts_by_fe(texts, fe_scores, n_select=10)
    assert len(result) == 3
