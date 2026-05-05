# halo_fep/benchmark/multimodal_world.py
"""Synthetic multimodal goal-inference world.

Hidden goal eta in {0, ..., n_hidden-1}. Each step produces:
  text_embed  ~ N(mu_text[eta],  0.1*I)   (d_model)
  image_embed ~ N(mu_image[eta], 0.3*I)   (d_model)

Returns tokens (n_tokens=2, d_model): [text_embed, image_embed].
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class MultimodalWorld(eqx.Module):
    mu_text:  jnp.ndarray  # (n_hidden, d_model)
    mu_image: jnp.ndarray  # (n_hidden, d_model)
    cfg: HaloFEPConfig = eqx.field(static=True)

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        self.mu_text  = jax.random.normal(k1, (cfg.n_hidden, cfg.d_model))
        self.mu_image = jax.random.normal(k2, (cfg.n_hidden, cfg.d_model))
        self.cfg = cfg

    def sample(self, eta: int, key: jnp.ndarray) -> tuple:
        """Sample one observation for goal eta.

        Returns:
            tokens: (n_tokens=2, d_model) — [text_embed, image_embed]
            eta: the goal index (passed through)
        """
        k1, k2 = jax.random.split(key)
        text_embed  = self.mu_text[eta]  + 0.1 * jax.random.normal(k1, (self.cfg.d_model,))
        image_embed = self.mu_image[eta] + 0.3 * jax.random.normal(k2, (self.cfg.d_model,))
        tokens = jnp.stack([text_embed, image_embed], axis=0)
        return tokens, eta
