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

    def __post_init__(self) -> None:
        assert self.n_agents % self.coarse_k == 0, "n_agents must be divisible by coarse_k"
        assert self.n_tokens >= 1, "n_tokens must be >= 1"
