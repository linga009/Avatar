import jax
import jax.numpy as jnp
import equinox as eqx
from fep_swarm.config import FEPConfig


class DiscreteGenerativeModel(eqx.Module):
    log_A: jnp.ndarray  # [n_obs, n_hidden]
    log_B: jnp.ndarray  # [n_hidden, n_hidden, n_actions]
    log_C: jnp.ndarray  # [n_obs]
    log_D: jnp.ndarray  # [n_hidden]

    def __init__(self, cfg: FEPConfig, key: jax.random.PRNGKey):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.log_A = jax.random.normal(k1, (cfg.n_obs, cfg.n_hidden))
        self.log_B = jax.random.normal(k2, (cfg.n_hidden, cfg.n_hidden, cfg.n_actions))
        self.log_C = jax.random.normal(k3, (cfg.n_obs,))
        self.log_D = jax.random.normal(k4, (cfg.n_hidden,))

    @property
    def A(self) -> jnp.ndarray:
        """P(s|η): column-stochastic [n_obs, n_hidden]"""
        return jax.nn.softmax(self.log_A, axis=0)

    @property
    def B(self) -> jnp.ndarray:
        """P(η'|η,a): column-stochastic per action [n_hidden, n_hidden, n_actions]"""
        return jax.nn.softmax(self.log_B, axis=0)

    @property
    def C(self) -> jnp.ndarray:
        """log prior preferences [n_obs]"""
        return jax.nn.log_softmax(self.log_C)

    @property
    def D(self) -> jnp.ndarray:
        """P(η) initial prior [n_hidden]"""
        return jax.nn.softmax(self.log_D)
