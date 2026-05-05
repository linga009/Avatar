import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class HoloEmbedding(eqx.Module):
    """Projects token embeddings into Poincaré half-space (x, z).

    x ∈ R^{d_boundary}: boundary position
    z ∈ (0, 1): radial depth (1=UV/deep, 0=IR/shallow)
    """

    x_proj: eqx.nn.Linear  # d_model -> d_boundary
    z_proj: eqx.nn.Linear  # d_model -> 1

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.x_proj = eqx.nn.Linear(cfg.d_model, cfg.d_boundary, key=k1)
        self.z_proj = eqx.nn.Linear(cfg.d_model, 1, key=k2)

    def __call__(self, h: jnp.ndarray):
        """Args:
            h: (N_tok, d_model)
        Returns:
            x: (N_tok, d_boundary)
            z: (N_tok, 1) in (0, 1)
        """
        x = jax.vmap(self.x_proj)(h)                    # (N_tok, d_boundary)
        z = jax.nn.sigmoid(jax.vmap(self.z_proj)(h))    # (N_tok, 1)
        return x, z
