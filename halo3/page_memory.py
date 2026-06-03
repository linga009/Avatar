"""PageCurveMemory — ring buffer with island eviction, compression, and echo."""
from __future__ import annotations
from typing import NamedTuple
import jax
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
    echo_gate_warmup: int = eqx.field(static=True)
    W_refine: jnp.ndarray
    g_echo_raw: jnp.ndarray

    def __init__(self, cfg: Halo3Config, key: jax.Array | None = None) -> None:
        self.max_cache = cfg.max_cache
        self.island_size = cfg.island_size
        self.d_model = cfg.d_model
        self.d_head = cfg.d_head
        self.echo_gate_warmup = cfg.echo_gate_warmup
        self.W_refine = jnp.zeros((cfg.d_model, cfg.d_model)) * 0.01
        self.g_echo_raw = jnp.array(-4.6)

    def init_state(self) -> PageMemState:
        return PageMemState(
            cache=jnp.zeros((self.max_cache, self.d_model)),
            n_cached=jnp.int32(0),
            island=jnp.zeros((self.island_size, self.d_model)),
            island_ptr=jnp.int32(0),
        )

    def compress_island(self, island: jnp.ndarray) -> jnp.ndarray:
        """Compress the island buffer into a single summary vector.

        Args:
            island: shape (island_size, d_model)

        Returns:
            summary: shape (d_model,)
        """
        summary = jnp.mean(island, axis=0)          # (d_model,)
        summary = summary + self.W_refine @ summary  # learned refinement
        return summary

    def is_island_full(self, state: PageMemState) -> bool:
        """Return True when the island write pointer has reached capacity."""
        return bool(state.island_ptr >= self.island_size)

    def apply_echo(
        self, state: PageMemState, summary: jnp.ndarray, tick: jnp.ndarray
    ) -> PageMemState:
        """Seed a freshly cleared island with a gated echo of the summary.

        The echo gate is clamped to [0, 0.1] during warmup and [0, 0.5] after.

        Args:
            state:   current PageMemState
            summary: shape (d_model,) — output of compress_island
            tick:    current training tick (used for warmup check)

        Returns:
            New PageMemState with island reset and island[0] = gated summary.
        """
        g_raw = jax.nn.sigmoid(self.g_echo_raw)
        in_warmup = tick < self.echo_gate_warmup
        g = jnp.where(in_warmup, jnp.clip(g_raw, 0.0, 0.1), jnp.clip(g_raw, 0.0, 0.5))
        faded = g * summary
        fresh_island = jnp.zeros((self.island_size, self.d_model))
        fresh_island = fresh_island.at[0].set(faded)
        return PageMemState(
            cache=state.cache,
            n_cached=state.n_cached,
            island=fresh_island,
            island_ptr=jnp.int32(1),
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
