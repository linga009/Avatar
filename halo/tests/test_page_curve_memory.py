# halo/tests/test_page_curve_memory.py
import torch
import pytest
from halo.config import HaloConfig
from halo.memory.page_curve_memory import PageCurveMemory


@pytest.fixture
def cfg():
    return HaloConfig(d_boundary=64, d_head=64, max_cache=4, island_size=2)


def test_add_within_capacity(cfg):
    mem = PageCurveMemory(cfg)
    for i in range(cfg.max_cache):
        x_i = torch.randn(cfg.d_boundary)
        attn_i = torch.softmax(torch.randn(cfg.max_cache), dim=0)
        kv_i = torch.randn(cfg.d_head)
        mem.add(x_i, attn_i, kv_i)
    assert len(mem.active_cache) == cfg.max_cache


def test_eviction_on_overflow(cfg):
    mem = PageCurveMemory(cfg)
    for i in range(cfg.max_cache + 1):
        x_i = torch.randn(cfg.d_boundary)
        attn_i = torch.softmax(torch.randn(i + 1), dim=0)[:cfg.max_cache]
        attn_i = attn_i / attn_i.sum()
        kv_i = torch.randn(cfg.d_head)
        mem.add(x_i, attn_i, kv_i)
    assert len(mem.active_cache) == cfg.max_cache
    assert len(mem.island_buffer) == 1


def test_island_size_cap(cfg):
    mem = PageCurveMemory(cfg)
    # Add max_cache + island_size + 5 tokens to fill island
    for i in range(cfg.max_cache + cfg.island_size + 5):
        x_i = torch.randn(cfg.d_boundary)
        attn_i = torch.softmax(torch.randn(cfg.max_cache), dim=0)
        kv_i = torch.randn(cfg.d_head)
        mem.add(x_i, attn_i, kv_i)
    assert len(mem.island_buffer) <= cfg.island_size


def test_generalized_entropy_positive(cfg):
    mem = PageCurveMemory(cfg)
    x_i = torch.randn(cfg.d_boundary)
    attn_i = torch.softmax(torch.randn(8), dim=0)
    s = mem.generalized_entropy(x_i, attn_i)
    assert s.item() > 0


def test_get_all_kv(cfg):
    mem = PageCurveMemory(cfg)
    for i in range(3):
        mem.add(torch.randn(cfg.d_boundary),
                torch.softmax(torch.randn(3), dim=0),
                torch.randn(cfg.d_head))
    kv = mem.get_all_kv()
    assert kv.shape == (3, cfg.d_head)


def test_reset_clears_state(cfg):
    mem = PageCurveMemory(cfg)
    for i in range(cfg.max_cache + 2):
        mem.add(torch.randn(cfg.d_boundary),
                torch.softmax(torch.randn(cfg.max_cache), dim=0),
                torch.randn(cfg.d_head))
    mem.reset()
    assert len(mem.active_cache) == 0
    assert len(mem.island_buffer) == 0
