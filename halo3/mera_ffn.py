"""MERA tensor train FFN — block-diagonal cores with bond mixing."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from halo3.config import Halo3Config


class MERAFFN(eqx.Module):
    """Tensor train FFN replacing a dense SwiGLU.

    Architecture
    ------------
    Standard SwiGLU uses three d_model×d_model projections (12 288 params for
    d_model=64).  This implementation decomposes each projection with a
    reduced hidden dimension (hidden_dim = d_model // 2) and block-diagonal
    tensor-train cores:

    * gate  : (d_model → hidden_dim)  split into n_cores cores of (chunk, chunk_h)
    * up    : (d_model → hidden_dim)  same shape
    * down  : (hidden_dim → d_model)  split into n_cores cores of (chunk_h, chunk)

    Bond vectors of shape (bond_dim,) carry cross-chunk information between
    adjacent cores.

    Parameter count (d_model=64, hidden_dim=32, n_cores=2, bond_dim=4)
    ------------------------------------------------------------------
    gate/up each: 2 × (32×16) + 1 × 4  =  1024 + 4  = 1028
    down        : 2 × (16×32) + 1 × 4  =  1024 + 4  = 1028
    Total       : 3 × 1028              = 3084

    Dense baseline: 3 × 64² = 12 288  →  MERA uses ~25 % (< 50 %).
    """

    gate_cores: list   # n_cores arrays of (chunk, chunk_h)
    gate_bonds: list   # (n_cores-1) arrays of (bond_dim,)
    up_cores: list
    up_bonds: list
    down_cores: list   # n_cores arrays of (chunk_h, chunk)
    down_bonds: list

    d_model: int = eqx.field(static=True)
    n_cores: int = eqx.field(static=True)
    chunk: int = eqx.field(static=True)       # d_model // n_cores
    chunk_h: int = eqx.field(static=True)     # hidden_dim // n_cores
    bond_dim: int = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key: jax.Array) -> None:
        self.d_model = cfg.d_model
        self.n_cores = cfg.mera_n_cores
        self.bond_dim = cfg.mera_bond_dim

        if cfg.d_model % cfg.mera_n_cores != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by "
                f"mera_n_cores ({cfg.mera_n_cores})"
            )

        hidden_dim = cfg.d_model // 2
        if hidden_dim % cfg.mera_n_cores != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"mera_n_cores ({cfg.mera_n_cores})"
            )

        self.chunk = cfg.d_model // cfg.mera_n_cores        # 32
        self.chunk_h = hidden_dim // cfg.mera_n_cores       # 16

        n_bond = cfg.mera_n_cores - 1

        def _make_enc(k: jax.Array):
            """Encoder cores: (chunk, chunk_h) + bond vectors (bond_dim,)."""
            keys = jax.random.split(k, cfg.mera_n_cores + n_bond)
            scale = 1.0 / jnp.sqrt(float(self.chunk))
            cores = [
                jax.random.normal(keys[i], (self.chunk, self.chunk_h)) * scale
                for i in range(cfg.mera_n_cores)
            ]
            bonds = [
                jax.random.normal(keys[cfg.mera_n_cores + i], (cfg.mera_bond_dim,))
                * (1.0 / jnp.sqrt(float(cfg.mera_bond_dim)))
                for i in range(n_bond)
            ]
            return cores, bonds

        def _make_dec(k: jax.Array):
            """Decoder cores: (chunk_h, chunk) + bond vectors (bond_dim,)."""
            keys = jax.random.split(k, cfg.mera_n_cores + n_bond)
            scale = 1.0 / jnp.sqrt(float(self.chunk_h))
            cores = [
                jax.random.normal(keys[i], (self.chunk_h, self.chunk)) * scale
                for i in range(cfg.mera_n_cores)
            ]
            bonds = [
                jax.random.normal(keys[cfg.mera_n_cores + i], (cfg.mera_bond_dim,))
                * (1.0 / jnp.sqrt(float(cfg.mera_bond_dim)))
                for i in range(n_bond)
            ]
            return cores, bonds

        k1, k2, k3 = jax.random.split(key, 3)
        self.gate_cores, self.gate_bonds = _make_enc(k1)
        self.up_cores, self.up_bonds = _make_enc(k2)
        self.down_cores, self.down_bonds = _make_dec(k3)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_enc(
        self,
        cores: list,
        bonds: list,
        x: jax.Array,
        in_chunk: int,
        out_chunk: int,
    ) -> jax.Array:
        """Apply encoder-style TT: x (d_model,) → hidden (hidden_dim,).

        Each bond vector additively gates its own slice of the next chunk's
        output via element-wise scaling of the first bond_dim elements.
        """
        chunks = [x[i * in_chunk : (i + 1) * in_chunk] for i in range(self.n_cores)]
        outputs = [chunks[i] @ cores[i] for i in range(self.n_cores)]
        for i, bond in enumerate(bonds):
            bd = self.bond_dim
            mix = outputs[i][:bd] * bond   # element-wise, shape (bond_dim,)
            outputs[i + 1] = outputs[i + 1].at[:bd].add(mix)
        return jnp.concatenate(outputs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, x: jax.Array) -> jax.Array:
        """SwiGLU-style forward pass.

        Args:
            x: input vector of shape (d_model,).

        Returns:
            output vector of shape (d_model,).
        """
        # gate/up project from d_model → hidden_dim
        gate = self._apply_enc(self.gate_cores, self.gate_bonds, x, self.chunk, self.chunk_h)
        up = self._apply_enc(self.up_cores, self.up_bonds, x, self.chunk, self.chunk_h)
        hidden = jax.nn.silu(gate) * up   # shape (hidden_dim,)
        # down project from hidden_dim → d_model
        return self._apply_enc(self.down_cores, self.down_bonds, hidden, self.chunk_h, self.chunk)
