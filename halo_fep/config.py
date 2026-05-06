from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HaloFEPConfig:
    # HALO dims
    d_model: int = 256
    d_boundary: int = 64
    n_heads: int = 4
    d_head: int = 64
    n_layers: int = 8
    d_state: int = 16
    d_ff: int = 512
    max_cache: int = 128
    island_size: int = 32
    flow_steps: int = 4
    delta_flow: float = 1.5
    bekenstein_alpha: float = 0.1
    lambda_bek: float = 0.1
    lambda_thermo: float = 0.05
    lambda_page: float = 0.05

    # FEP dims
    n_hidden: int = 8
    n_obs: int = 4
    n_actions: int = 4
    n_policies: int = 8
    tau: int = 3
    inf_steps: int = 16
    inf_lr: float = 0.01
    beta: float = 1.0

    # Swarm
    n_agents: int = 256
    kappa: float = 0.3
    topology: Literal["all2all", "sparse", "grid"] = "all2all"
    coarse_k: int = 16

    # Bridge
    n_tokens: int = 2   # number of HALO tokens (1 text + 1 image for benchmark)

    # Joint training
    lambda_fep: float = 0.1
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42

    # Heartbeat
    wake_threshold: float = 2.5   # FE above this triggers LLM wake cycle
    tick_interval:  int   = 60    # seconds between subconscious ticks

    def __post_init__(self) -> None:
        if self.n_agents % self.coarse_k != 0:
            raise ValueError(f"n_agents ({self.n_agents}) must be divisible by coarse_k ({self.coarse_k})")
        if self.n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {self.n_tokens}")
        if self.wake_threshold <= 0.0:
            raise ValueError(f"wake_threshold must be > 0, got {self.wake_threshold}")
        if self.tick_interval <= 0:
            raise ValueError(f"tick_interval must be > 0, got {self.tick_interval}")
