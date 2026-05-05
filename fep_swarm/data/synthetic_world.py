import jax
import jax.numpy as jnp
from fep_swarm.config import FEPConfig


def make_tmaze(cfg: FEPConfig) -> tuple:
    """
    T-maze generative model.
    States: 0=start, 1=left_arm, 2=right_arm, 3+=other
    Obs:    0=center, 1=left, 2=right, 3=reward, others=neutral
    Returns (A, B, C, D) as jnp arrays with cfg dimensions.
    """
    n_h, n_o, n_a = cfg.n_hidden, cfg.n_obs, cfg.n_actions

    # A: diagonal-dominant likelihood (each state mostly produces its obs)
    A = jnp.eye(n_o, n_h) * 0.85 + 0.15 / n_o
    A = A / A.sum(axis=0, keepdims=True)

    # B: transition matrix per action
    B = jnp.zeros((n_h, n_h, n_a))
    # action 0: stay
    B = B.at[:, :, 0].set(jnp.eye(n_h))
    # action 1: go left (state 0 -> 1)
    B_left = jnp.eye(n_h).at[min(1, n_h - 1), 0].set(1.0).at[0, 0].set(0.0)
    B = B.at[:, :, 1].set(B_left)
    # action 2: go right (state 0 -> 2)
    B_right = jnp.eye(n_h).at[min(2, n_h - 1), 0].set(1.0).at[0, 0].set(0.0)
    B = B.at[:, :, 2].set(B_right)
    # action 3: go back (1 -> 0, 2 -> 0)
    B_back = jnp.eye(n_h)
    for src in [1, 2]:
        if src < n_h:
            B_back = B_back.at[:, src].set(0.0).at[0, src].set(1.0)
    B = B.at[:, :, 3].set(B_back)

    # C: prefer reward observation (obs index 3 if available)
    C = jnp.zeros(n_o).at[min(3, n_o - 1)].set(2.0)

    # D: start at state 0
    D = jax.nn.one_hot(0, n_h)

    return A, B, C, D


def sample_obs(
    eta_onehot: jnp.ndarray,
    A_true: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> int:
    """Sample observation index from P(o|eta)."""
    obs_probs = A_true @ eta_onehot
    return int(jax.random.choice(key, obs_probs.shape[0], p=obs_probs))


def transition_world(
    eta_onehot: jnp.ndarray,
    action_onehot: jnp.ndarray,
    B_true: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Apply action to world, return new eta one-hot."""
    a = jnp.argmax(action_onehot)
    eta_probs = B_true[:, :, a] @ eta_onehot
    eta_idx = jax.random.choice(key, eta_probs.shape[0], p=eta_probs)
    return jax.nn.one_hot(eta_idx, eta_onehot.shape[0])
