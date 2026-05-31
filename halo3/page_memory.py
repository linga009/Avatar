"""PageCurveMemory — ring buffer with island eviction."""
from __future__ import annotations
from typing import NamedTuple
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config


class PageMemState(NamedTuple):
    cache: jnp.ndarray
    n_cached: jnp.ndarray
    island: jnp.ndarray
    island_ptr: jnp.ndarray


class PageCurveMemory(eqx.Module):
    max_cache: int = eqx.field(static=True)
    island_size: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config) -> None:
        self.max_cache = cfg.max_cache
        self.island_size = cfg.island_size
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head

    def init_state(self) -> PageMemState:
        return PageMemState(
            cache=jnp.zeros((self.max_cache, self.d_model)),
            n_cached=jnp.int32(0),
            island=jnp.zeros((self.island_size, self.d_model)),
            island_ptr=jnp.int32(0),
        )

    def __call__(self, x_i, state):
        write_ptr = jnp.int32(state.n_cached % self.max_cache)
        new_cache = state.cache.at[write_ptr].set(x_i)
        is_full = state.n_cached >= self.max_cache
        sq = jnp.sum(new_cache ** 2, axis=-1)
        qr = jnp.sum(new_cache ** 4, axis=-1)
        pr = sq ** 2 / (qr + 1e-12)
        s_gen = sq * pr
        evict_idx = jnp.argmin(s_gen)
        evicted = new_cache[evict_idx]
        iptr = jnp.int32(state.island_ptr % self.island_size)
        new_island = jnp.where(is_full, state.island.at[iptr].set(evicted), state.island)
        new_iptr = jnp.where(is_full, jnp.int32(state.island_ptr + 1), state.island_ptr)
        return PageMemState(cache=new_cache, n_cached=jnp.int32(state.n_cached + 1), island=new_island, island_ptr=new_iptr)
