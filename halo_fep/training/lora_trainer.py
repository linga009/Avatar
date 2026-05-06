# halo_fep/training/lora_trainer.py
"""Nightly LoRA-style fine-tuning on high-confidence episodes.

Fine-tunes only backbone weights (SSM + attention projections) via
eqx.filter_grad. Other weights (bridges, gm, embedder) are frozen.
Reverts if loss increases after training.

Usage:
    trainer = LoRATrainer(cfg, n_steps=100, lr=1e-4)
    model, log = trainer.run(model, high_confidence_episodes)
"""
from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax
import numpy as np

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.loss import unified_elbo_loss
from halo_fep.memory.schema import Episode

log = logging.getLogger(__name__)


def _backbone_filter(model: HaloFEPModel):
    """Return a boolean pytree: True only for backbone leaves."""
    false_model = jtu.tree_map(lambda _: False, model)
    true_backbone = jtu.tree_map(lambda _: True, model.backbone)
    return eqx.tree_at(lambda m: m.backbone, false_model, true_backbone)


class LoRATrainer:
    def __init__(
        self,
        cfg: HaloFEPConfig,
        n_steps: int = 100,
        lr: float = 1e-4,
    ) -> None:
        self.cfg     = cfg
        self.n_steps = n_steps
        self.opt     = optax.adam(lr)

    def run(
        self,
        model: HaloFEPModel,
        episodes: list[Episode],
    ) -> tuple[HaloFEPModel, dict[str, Any]]:
        """Fine-tune on episodes. Returns (model, log_dict)."""
        if not episodes:
            return model, {"loss_before": 0.0, "loss_after": 0.0, "n_episodes": 0}

        key   = jax.random.PRNGKey(self.cfg.seed)
        carry = model.init_carry(key)

        # Measure loss before training
        loss_before = self._mean_loss(model, carry, episodes, key)
        log.info(f"LoRA fine-tune: loss_before={loss_before:.4f}, n_episodes={len(episodes)}")

        checkpoint = model
        opt_state  = self.opt.init(eqx.filter(model, eqx.is_array))

        for step in range(self.n_steps):
            # Sample episode for this step (cycle through)
            ep_idx  = step % len(episodes)
            tokens  = jnp.array(episodes[ep_idx].tokens)
            key, subkey = jax.random.split(key)

            (loss, _), grads = eqx.filter_value_and_grad(unified_elbo_loss, has_aux=True)(
                model, carry, tokens, subkey
            )

            # Zero out grads outside backbone
            filter_mask = _backbone_filter(model)
            grads = jtu.tree_map(
                lambda g, mask: g if mask else jnp.zeros_like(g),
                grads,
                filter_mask,
            )

            updates, opt_state = self.opt.update(
                eqx.filter(grads, eqx.is_array),
                opt_state,
                eqx.filter(model, eqx.is_array),
            )
            model = eqx.apply_updates(model, updates)

            carry, _ = halo_fep_step(model, carry, tokens, subkey)

        loss_after = self._mean_loss(model, carry, episodes, key)
        log.info(f"LoRA fine-tune: loss_after={loss_after:.4f}")

        if loss_after > loss_before:
            log.warning("Loss increased after fine-tuning — reverting to checkpoint.")
            model = checkpoint

        return model, {
            "loss_before": float(loss_before),
            "loss_after":  float(loss_after),
            "n_episodes":  len(episodes),
        }

    def _mean_loss(
        self,
        model: HaloFEPModel,
        carry,
        episodes: list[Episode],
        key: jnp.ndarray,
    ) -> float:
        losses = []
        for ep in episodes[:10]:  # cap evaluation at 10 for speed
            tokens = jnp.array(ep.tokens)
            key, subkey = jax.random.split(key)
            loss, _ = unified_elbo_loss(model, carry, tokens, subkey)
            losses.append(float(loss))
        return float(np.mean(losses))
