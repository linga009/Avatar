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


def test_update_llm_output():
    with tempfile.TemporaryDirectory() as d:
        store = EpisodeStore(path=d)
        ep = make_episode()
        store.add(ep)
        store.update_llm_output(ep.id, "SEARCH: test query")
        results = store.retrieve(np.random.randn(256).astype(np.float32), k=1)
        assert results[0].llm_output == "SEARCH: test query"


def test_get_prioritized_returns_correct_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = EpisodeStore(tmp)
        qe = np.random.randn(256).astype(np.float32)
        qe /= np.linalg.norm(qe) + 1e-8
        for i in range(10):
            ep = Episode(
                query=f"q{i}",
                tokens=np.zeros((32, 256), dtype=np.float32),
                swarm_mu=np.zeros((256, 8), dtype=np.float32),
                free_energy=1.0,
                free_energy_delta=-(0.05 + i * 0.01),
            )
            store.add(ep, query_embed=qe.copy())
        episodes, weights = store.get_prioritized(n=5, alpha=0.6, beta=0.4)
        assert len(episodes) == 5
        assert weights.shape == (5,)
        assert np.all(weights > 0)
        assert np.max(weights) <= 1.0


def test_get_prioritized_fewer_than_n():
    with tempfile.TemporaryDirectory() as tmp:
        store = EpisodeStore(tmp)
        qe = np.ones(256, dtype=np.float32)
        qe /= np.linalg.norm(qe)
        for delta in [-0.1, -0.2]:
            ep = Episode(
                query="q",
                tokens=np.zeros((32, 256), dtype=np.float32),
                swarm_mu=np.zeros((256, 8), dtype=np.float32),
                free_energy=1.0,
                free_energy_delta=delta,
            )
            store.add(ep, query_embed=qe.copy())
        episodes, weights = store.get_prioritized(n=10, alpha=0.6, beta=0.4)
        assert len(episodes) == 2


def test_get_prioritized_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = EpisodeStore(tmp)
        episodes, weights = store.get_prioritized(n=5, alpha=0.6, beta=0.4)
        assert episodes == []
        assert len(weights) == 0
