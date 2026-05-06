# halo_fep/training/mesu.py
"""MESU — Metaplasticity from Synaptic Uncertainty optimizer.

Implements the boundary-free continual learning update rule from:
  "Bayesian continual learning and forgetting in neural networks"
  Nature Communications, 2025.

Each parameter's learning rate is scaled by its uncertainty (sigma):
    theta <- theta - lr * grad / (sigma + epsilon)
    sigma <- sigma + eta * (grad^2 - sigma)

High-gradient parameters accumulate high sigma (high certainty), which
automatically reduces their future learning rate — preventing forgetting
of well-learned patterns without explicit task boundaries.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.tree_util as jtu
import optax


class MESUState(NamedTuple):
    """Optimizer state: per-parameter uncertainty estimates."""
    sigma: Any  # PyTree matching param structure, float32


def mesu(
    lr: float = 1e-4,
    eta: float = 0.01,
    epsilon: float = 1e-8,
) -> optax.GradientTransformation:
    """Create a MESU gradient transformation.

    Args:
        lr: Global learning rate.
        eta: Uncertainty update rate. Controls how fast sigma adapts to
             gradient variance. Typical range: [0.001, 0.1].
        epsilon: Numerical stability constant added to sigma denominator.

    Returns:
        An optax.GradientTransformation compatible with all optax utilities.
    """
    def init_fn(params: Any) -> MESUState:
        # Initialize sigma to 1 (maximum uncertainty / uninformative prior)
        return MESUState(sigma=jtu.tree_map(jnp.ones_like, params))

    def update_fn(
        updates: Any,
        state: MESUState,
        params: Any = None,
    ) -> tuple[Any, MESUState]:
        sigma = state.sigma

        # Scale updates by inverse uncertainty: high sigma = small update
        scaled_updates = jtu.tree_map(
            lambda g, s: -lr * g / (s + epsilon),
            updates,
            sigma,
        )

        # Update uncertainty via gradient variance EMA:
        # sigma converges toward E[g^2] (the expected squared gradient)
        new_sigma = jtu.tree_map(
            lambda s, g: jnp.clip(s + eta * (g ** 2 - s), 1e-8, 1e6),
            sigma,
            updates,
        )

        return scaled_updates, MESUState(sigma=new_sigma)

    return optax.GradientTransformation(init_fn, update_fn)
