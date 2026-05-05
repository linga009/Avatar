from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FEPConfig:
    # Generative model
    n_hidden: int = 8          # |η| discrete hidden state dimension
    n_obs: int = 4             # |s| discrete observation dimension
    n_actions: int = 4         # |a| action space size
    n_policies: int = 8        # candidate policies π to evaluate for EFE
    tau: int = 3               # planning horizon (timesteps ahead)

    # Continuous likelihood NeuralODE
    obs_dim: int = 16          # continuous observation embedding dimension
    ode_width: int = 64        # MLP hidden width for vector field f_θ
    ode_depth: int = 2         # MLP depth

    # Agent belief inference
    inf_steps: int = 16        # gradient descent steps per belief update
    inf_lr: float = 0.01       # belief update step size (applied to logits)
    beta: float = 1.0          # policy temperature softmax(−β·G)

    # Swarm
    n_agents: int = 256        # N agents (vmapped, zero Python loops)
    kappa: float = 0.3         # coupling strength κ ∈ [0, 1]
    topology: Literal["all2all", "sparse", "grid"] = "all2all"
    sparse_p: float = 0.1      # edge probability for sparse topology

    # Macro
    coarse_k: int = 16         # agents per coarse-grain group (must divide n_agents)
    eig_gap: float = 10.0      # |λ_micro|/|λ_macro| proof threshold

    # Training
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42

    def __post_init__(self):
        assert self.n_agents % self.coarse_k == 0, (
            f"n_agents ({self.n_agents}) must be divisible by coarse_k ({self.coarse_k})"
        )
