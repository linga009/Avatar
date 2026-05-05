from typing import NamedTuple
import jax
import jax.numpy as jnp
from fep_swarm.config import FEPConfig


class EnvState(NamedTuple):
    eta: jnp.ndarray  # [n_hidden] true hidden world state (one-hot)
    step: int


def init_env(cfg: FEPConfig, key: jax.random.PRNGKey) -> EnvState:
    eta = jax.nn.one_hot(
        jax.random.randint(key, (), 0, cfg.n_hidden), cfg.n_hidden
    )
    return EnvState(eta=eta, step=0)


def observe(
    env: EnvState,
    cfg: FEPConfig,
    A_true: jnp.ndarray,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Each of N agents independently samples an observation from P(s|eta).
    Returns obs_indices: Array[N] integer observation indices."""
    obs_probs = A_true @ env.eta  # [n_obs]
    keys = jax.random.split(key, cfg.n_agents)
    return jax.vmap(
        lambda k: jax.random.choice(k, cfg.n_obs, p=obs_probs)
    )(keys)


def step_env(
    env: EnvState,
    actions: jnp.ndarray,   # [N, n_actions]
    B_true: jnp.ndarray,
    cfg: FEPConfig,
    key: jax.random.PRNGKey,
) -> EnvState:
    """Transition world via mean agent action."""
    a_idx = jnp.argmax(actions.mean(axis=0))
    eta_probs = B_true[:, :, a_idx] @ env.eta
    eta_new = jax.nn.one_hot(
        jax.random.choice(key, cfg.n_hidden, p=eta_probs), cfg.n_hidden
    )
    return EnvState(eta=eta_new, step=env.step + 1)
