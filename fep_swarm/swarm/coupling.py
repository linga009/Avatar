import jax
import jax.numpy as jnp
from fep_swarm.config import FEPConfig


def build_coupling_matrix(cfg: FEPConfig, key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Build N×N coupling matrix W (W[i,j] = influence of agent j on agent i).
    Topologies: "all2all" | "sparse" | "grid"
    """
    N = cfg.n_agents
    if cfg.topology == "all2all":
        return jnp.ones((N, N)) / N

    elif cfg.topology == "sparse":
        mask = jax.random.bernoulli(key, cfg.sparse_p, (N, N))
        mask = mask & ~jnp.eye(N, dtype=bool)  # remove self-loops
        row_sums = mask.sum(axis=1, keepdims=True).astype(float) + 1e-8
        return mask.astype(float) / row_sums

    elif cfg.topology == "grid":
        side = int(N ** 0.5)
        assert side * side == N, f"n_agents must be perfect square for grid, got {N}"
        offsets = jnp.array([-side, side, -1, 1])
        idx = jnp.arange(N)
        W = jnp.zeros((N, N))
        for off in offsets:
            j = idx + off
            valid = (j >= 0) & (j < N)
            if off == -1:
                valid = valid & (idx % side != 0)
            if off == 1:
                valid = valid & (idx % side != side - 1)
            j_clipped = jnp.clip(j, 0, N - 1)
            W = W + jnp.where(valid[:, None] & (jnp.arange(N) == j_clipped[:, None]), 1.0, 0.0)
        row_sums = W.sum(axis=1, keepdims=True) + 1e-8
        return W / row_sums

    else:
        raise ValueError(f"Unknown topology: {cfg.topology!r}")


def apply_coupling(
    obs_self: jnp.ndarray,   # [N, n_obs] agent own observations (probabilities)
    actions: jnp.ndarray,    # [N, n_actions] agent actions (probabilities)
    W: jnp.ndarray,          # [N, N]
    cfg: FEPConfig,
) -> jnp.ndarray:
    """
    obs_new[i] = (1−κ)·obs_self[i] + κ·Σ_j W[i,j]·action_proj[j]
    Projects actions to obs dimension via truncation/tiling + softmax.
    """
    n_obs = obs_self.shape[-1]
    n_a = actions.shape[-1]
    if n_a >= n_obs:
        action_proj = actions[:, :n_obs]
    else:
        reps = (n_obs + n_a - 1) // n_a
        action_proj = jnp.tile(actions, (1, reps))[:, :n_obs]
    action_proj = jax.nn.softmax(action_proj, axis=-1)

    influence = W @ action_proj  # [N, n_obs]
    return (1.0 - cfg.kappa) * obs_self + cfg.kappa * influence
