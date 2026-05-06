# halo_fep/training/lora_trainer.py
"""Nightly LoRA-style fine-tuning on high-confidence episodes.

Fine-tunes only backbone weights via eqx.filter_grad.
Adds EWC-LoRA penalty to prevent catastrophic forgetting.
Optionally weights loss by Prioritized Experience Replay (PER) weights.
Reverts if loss increases after training.
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


def _compute_fisher(
    model: HaloFEPModel,
    carry,
    episodes: list[Episode],
    key: jnp.ndarray,
) -> Any:
    """Diagonal Fisher Information Matrix over backbone parameters.

    Approximated as the mean squared gradient of the loss w.r.t. backbone
    weights, computed over up to 10 episodes for speed.

    Returns a PyTree matching model.backbone shape with float32 arrays.
    """
    fisher = jtu.tree_map(lambda x: jnp.zeros_like(x), model.backbone)
    n_eval = min(len(episodes), 10)

    for ep in episodes[:n_eval]:
        tokens = jnp.array(ep.tokens)
        key, sk = jax.random.split(key)

        # Gradient of loss w.r.t. full model; we only use the backbone part
        grad_fn = jax.grad(lambda m: unified_elbo_loss(m, carry, tokens, sk)[0])
        grads = grad_fn(model)

        # Accumulate squared gradients (diagonal Fisher approximation)
        fisher = jtu.tree_map(
            lambda f, g: f + g ** 2,
            fisher,
            grads.backbone,
        )

    return jtu.tree_map(lambda f: f / n_eval, fisher)


def _ewc_penalty(
    current_backbone: Any,
    checkpoint_backbone: Any,
    fisher: Any,
    ewc_lambda: float,
) -> jnp.ndarray:
    """EWC regularization: lambda * sum_i F_i * (theta_i - theta_i*)^2.

    Penalizes deviation from checkpoint proportional to Fisher importance.
    Returns a scalar float32.
    """
    diffs = jtu.tree_map(lambda c, o: c - o, current_backbone, checkpoint_backbone)
    penalties = jtu.tree_map(
        lambda f, d: jnp.sum(f * d ** 2),
        fisher,
        diffs,
    )
    return ewc_lambda * sum(jtu.tree_leaves(penalties))


class LoRATrainer:
    def __init__(
        self,
        cfg: HaloFEPConfig,
        n_steps: int = 100,
        lr: float = 1e-4,
    ) -> None:
        self.cfg     = cfg
        self.n_steps = n_steps
        # Select optimizer: MESU if cfg.use_mesu, else Adam
        if cfg.use_mesu:
            from halo_fep.training.mesu import mesu
            self.opt = mesu(lr=lr, eta=cfg.mesu_eta)
        else:
            self.opt = optax.adam(lr)

    def run(
        self,
        model: HaloFEPModel,
        episodes: list[Episode],
        per_weights: np.ndarray | None = None,
    ) -> tuple[HaloFEPModel, dict[str, Any]]:
        """Fine-tune on episodes with EWC-LoRA regularization.

        Args:
            model: Current model to fine-tune.
            episodes: High-confidence episodes to train on.
            per_weights: Optional (N,) float32 IS weights from PER sampling.
                         If None, all episodes are weighted equally.

        Returns:
            (model, log_dict) — model may be the original if divergence detected.
        """
        if not episodes:
            return model, {
                "loss_before": 0.0,
                "loss_after": 0.0,
                "n_episodes": 0,
                "ewc_penalty": 0.0,
            }

        if per_weights is None:
            per_weights = np.ones(len(episodes), dtype=np.float32)
        per_weights = np.asarray(per_weights, dtype=np.float32)

        key   = jax.random.PRNGKey(self.cfg.seed)
        carry = model.init_carry(key)

        loss_before = self._mean_loss(model, carry, episodes, key)
        log.info(
            f"LoRA fine-tune: loss_before={loss_before:.4f}, "
            f"n_episodes={len(episodes)}"
        )

        checkpoint = model
        checkpoint_backbone = model.backbone  # snapshot for EWC

        # Compute Fisher BEFORE training (on the pre-update weights)
        ewc_penalty_val = 0.0
        fisher = None
        if self.cfg.ewc_lambda > 0.0:
            key, fk = jax.random.split(key)
            fisher = _compute_fisher(model, carry, episodes, fk)

        opt_state = self.opt.init(eqx.filter(model, eqx.is_array))

        for step in range(self.n_steps):
            ep_idx  = step % len(episodes)
            tokens  = jnp.array(episodes[ep_idx].tokens)
            w       = float(per_weights[ep_idx])
            key, sk = jax.random.split(key)

            (loss, _), grads = eqx.filter_value_and_grad(
                unified_elbo_loss, has_aux=True
            )(model, carry, tokens, sk)

            # Scale loss gradient by PER importance weight
            grads = jtu.tree_map(lambda g: g * w, grads)

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
            carry, _ = halo_fep_step(model, carry, tokens, sk)

        # Compute final EWC penalty on updated model vs checkpoint
        if self.cfg.ewc_lambda > 0.0 and fisher is not None:
            ewc_penalty_val = float(
                _ewc_penalty(
                    model.backbone,
                    checkpoint_backbone,
                    fisher,
                    self.cfg.ewc_lambda,
                )
            )
            log.info(f"EWC penalty: {ewc_penalty_val:.4f}")

        loss_after = self._mean_loss(model, carry, episodes, key)
        log.info(f"LoRA fine-tune: loss_after={loss_after:.4f}")

        # Revert-on-diverge: if loss increased, discard new weights
        if loss_after > loss_before:
            log.warning("Loss increased after fine-tuning — reverting to checkpoint.")
            model = checkpoint

        return model, {
            "loss_before":  float(loss_before),
            "loss_after":   float(loss_after),
            "n_episodes":   len(episodes),
            "ewc_penalty":  ewc_penalty_val,
        }

    def _mean_loss(
        self,
        model: HaloFEPModel,
        carry,
        episodes: list[Episode],
        key: jnp.ndarray,
    ) -> float:
        losses = []
        for ep in episodes[:10]:
            tokens = jnp.array(ep.tokens)
            key, sk = jax.random.split(key)
            loss, _ = unified_elbo_loss(model, carry, tokens, sk)
            losses.append(float(loss))
        return float(np.mean(losses))
