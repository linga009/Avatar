# halo_fep/halo_jax/simple_ssm.py
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class SimpleSSM(eqx.Module):
    """Diagonal-state linear SSM (simplified Mamba without selective scan).

    h_t = exp(A) ⊙ h_{t-1} + B·x_t
    y_t = C·h_t + D ⊙ x_t

    Sequential scan over the token dimension using jax.lax.scan.
    """

    A: jnp.ndarray          # (d_state,) diagonal state matrix (log scale)
    B: eqx.nn.Linear        # d_model -> d_state
    C: eqx.nn.Linear        # d_state -> d_model
    D: jnp.ndarray          # (d_model,) skip connection

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.A = jnp.full((cfg.d_state,), -1.0)   # init: stable decay
        self.B = eqx.nn.Linear(cfg.d_model, cfg.d_state, use_bias=False, key=k1)
        self.C = eqx.nn.Linear(cfg.d_state, cfg.d_model, use_bias=False, key=k2)
        self.D = jnp.ones(cfg.d_model)

    def __call__(self, xs: jnp.ndarray) -> jnp.ndarray:
        """Args:
            xs: (seq_len, d_model)
        Returns:
            ys: (seq_len, d_model)
        """
        def step(h: jnp.ndarray, x_t: jnp.ndarray):
            h_new = jnp.exp(self.A) * h + self.B(x_t)
            y_t   = self.C(h_new) + self.D * x_t
            return h_new, y_t

        h0 = jnp.zeros(self.A.shape[0])
        _, ys = jax.lax.scan(step, h0, xs)
        return ys
