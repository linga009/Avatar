from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Halo3Config:
    """Immutable configuration for HoloBiont 3.0 Physics Engine."""

    # Backbone — empirically fitted for 6 GB GPU (GTX 1660 Ti)
    d_model: int = 2048
    d_boundary: int = 64
    n_heads: int = 16
    d_head: int = 128
    n_layers: int = 48
    d_state: int = 128
    layer_pattern: str = "SSSSSH"
    n_shared_attn: int = 2
    lora_rank: int = 16
    reversible: bool = True

    # MERA-FFN
    mera_bond_dim: int = 64
    mera_n_cores: int = 4

    # Lorentz
    init_curvature: float = 1.0

    # Hamiltonian
    n_leapfrog_steps: int = 3
    leapfrog_step_size: float = 0.1
    lambda_energy: float = 0.1

    # Kuramoto
    n_clusters: int = 32
    n_hidden: int = 16
    kuramoto_dt: float = 0.1
    init_coupling: float = 1.0
    lambda_sync: float = 0.01

    # Page memory
    max_cache: int = 128
    island_size: int = 32

    # Training
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42
    galore_rank: int = 64
    lisa_active_layers: int = 2

    # Bridges
    n_tokens: int = 32
    n_obs: int = 8
    n_actions: int = 8

    # Meta-layer
    meta_n_hidden: int = 8
    meta_n_actions: int = 4
    meta_k: int = 10

    # Homeostatic
    homeo_sync_threshold: float = 0.6
    homeo_blend_clip: float = 1.0

    # Heartbeat
    tick_interval: int = 60

    def __post_init__(self) -> None:
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * d_head ({self.d_head}) "
                f"must equal d_model ({self.d_model})"
            )
        if self.n_layers % len(self.layer_pattern) != 0:
            raise ValueError(
                f"n_layers ({self.n_layers}) must be divisible by "
                f"len(layer_pattern) ({len(self.layer_pattern)})"
            )
        if self.n_leapfrog_steps < 1:
            raise ValueError(f"n_leapfrog_steps must be >= 1")
        if self.meta_k < 1:
            raise ValueError(f"meta_k must be >= 1")
