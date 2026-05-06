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
