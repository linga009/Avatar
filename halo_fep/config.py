from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HaloFEPConfig:
    """Immutable configuration for the HoloBiont Persistent Mind system.

    All architectural dimensions, hyper-parameters, and operational settings
    live here.  The ``__post_init__`` method validates that the chosen values
    are mutually consistent, catching shape-mismatch bugs before any array is
    allocated.

    HALO backbone
    -------------
    d_model     : Total hidden dimension (must equal n_heads * d_head).
    d_boundary  : Dimension of the Poincaré boundary embedding.
    n_heads     : Number of attention heads.
    d_head      : Dimension per head (d_model // n_heads).
    n_layers    : Number of stacked HALO layers.
    d_state     : SSM state dimension.
    d_ff        : Feed-forward hidden size.
    max_cache   : Recurrent state cache length.
    island_size : Local processing island size.
    flow_steps  : Continuous flow-matching steps (must be ≤ n_layers).
    delta_flow  : Flow step-size scaling factor.

    Bekenstein / physics regularisation
    ------------------------------------
    bekenstein_alpha : Attention entropy budget coefficient.
    lambda_bek       : Weight of Bekenstein loss term.
    lambda_thermo    : Weight of thermodynamic entropy-production term.
    lambda_page      : Weight of Page-curve alignment term.

    FEP generative model
    --------------------
    n_hidden   : Number of discrete hidden states η.
    n_obs      : Observation dimensionality (must be ≥ 1).
    n_actions  : Action space size.
    n_policies : Number of candidate policies evaluated per step.
    tau        : Planning horizon (time steps).
    inf_steps  : Variational inference iterations per tick.
    inf_lr     : Belief update learning rate.
    beta       : Inverse temperature for softmax over Expected Free Energy.

    Swarm
    -----
    n_agents : Number of parallel agents (must be divisible by coarse_k).
    kappa    : Lateral inhibition / interaction strength between agents.
    topology : Communication graph topology.
    coarse_k : Coarse-graining factor for swarm belief aggregation.

    Bridge
    ------
    n_tokens : Number of HALO tokens fed per tick (text + image slots).
               **Default is 32** — the production heartbeat value.  The old
               benchmark default of 2 was a source of silent shape mismatches
               and has been retired.

    Joint training
    --------------
    lambda_fep : FEP loss weight in joint ELBO.
    lr         : Adam learning rate for bootstrap / joint training.
    n_steps    : Training steps for joint optimisation.
    seed       : Global random seed.

    Continual learning
    ------------------
    ewc_lambda : EWC penalty weight (0 disables EWC).
    per_alpha  : PER priority exponent (0 = uniform, 1 = full priority).
    per_beta   : PER importance-sampling correction exponent.
    use_mesu   : If True, use MESU optimizer instead of Adam for nightly LoRA.
    mesu_eta   : MESU uncertainty EMA rate.

    Heartbeat
    ---------
    wake_threshold : Free-energy level (nats) above which the LLM is loaded.
    tick_interval  : Seconds between subconscious ticks.
    """

    # HALO dims
    d_model: int = 1024
    d_boundary: int = 64
    n_heads: int = 16
    d_head: int = 64
    n_layers: int = 12
    d_state: int = 16
    d_ff: int = 4096
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

    # Bridge — IMPORTANT: n_tokens=32 is the production default.
    # The previous default of 2 was a benchmark-only shortcut that caused
    # silent (N_agents, 2) vs (32, d_model) shape mismatches at runtime.
    n_tokens: int = 32

    # Joint training
    lambda_fep: float = 0.1
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42

    # Continual learning
    ewc_lambda:  float = 0.1    # EWC penalty weight (0 = disabled)
    per_alpha:   float = 0.6    # PER priority exponent
    per_beta:    float = 0.4    # PER importance-sampling correction exponent
    use_mesu:    bool  = False   # Use MESU optimizer instead of Adam for nightly LoRA
    mesu_eta:    float = 0.01   # MESU uncertainty EMA rate

    # Heartbeat
    wake_threshold: float = 2.5   # FE above this triggers LLM wake cycle
    tick_interval:  int   = 60    # seconds between subconscious ticks

    def __post_init__(self) -> None:
        """Validate mutual consistency of configuration values.

        Raises ValueError for any dimension or parameter that would cause a
        silent shape mismatch or numerical failure at runtime.
        """
        # --- Swarm ---
        if self.n_agents % self.coarse_k != 0:
            raise ValueError(
                f"n_agents ({self.n_agents}) must be divisible by coarse_k ({self.coarse_k})"
            )

        # --- Attention head dimensions ---
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * d_head ({self.d_head}) must equal "
                f"d_model ({self.d_model}), got {self.n_heads * self.d_head}"
            )

        # --- Flow steps vs layers ---
        if self.flow_steps > self.n_layers:
            raise ValueError(
                f"flow_steps ({self.flow_steps}) must be <= n_layers ({self.n_layers})"
            )

        # --- Token count ---
        if self.n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {self.n_tokens}")

        # --- Heartbeat ---
        if self.wake_threshold <= 0.0:
            raise ValueError(f"wake_threshold must be > 0, got {self.wake_threshold}")
        if self.tick_interval <= 0:
            raise ValueError(f"tick_interval must be > 0, got {self.tick_interval}")

        # --- FEP ---
        if self.n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {self.n_hidden}")
        if self.n_obs < 1:
            raise ValueError(f"n_obs must be >= 1, got {self.n_obs}")
        if self.n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {self.n_actions}")
        if self.tau < 1:
            raise ValueError(f"tau must be >= 1, got {self.tau}")

        # --- Learning rates / weights ---
        if not (0.0 <= self.ewc_lambda):
            raise ValueError(f"ewc_lambda must be >= 0, got {self.ewc_lambda}")
        if not (0.0 <= self.per_alpha <= 1.0):
            raise ValueError(f"per_alpha must be in [0, 1], got {self.per_alpha}")
        if not (0.0 <= self.per_beta <= 1.0):
            raise ValueError(f"per_beta must be in [0, 1], got {self.per_beta}")
