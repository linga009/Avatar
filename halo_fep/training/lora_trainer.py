# halo_fep/training/lora_trainer.py
"""Nightly LoRA-style fine-tuning on high-confidence episodes.

Fixes applied in this version
------------------------------
1. **Carry contamination (Bug 2)**: The training loop previously accumulated
   a single carry across all 100 steps, cycling through episodes.  The carry
   from episode i was passed unmodified into episode i+1, creating an
   uncontrolled stateful dependency.  Each episode step now re-initialises
   the carry from ``model.init_carry()`` to guarantee independent evaluation.

2. **Loss evaluation truncation (Bug 4)**: ``_mean_loss`` previously evaluated
   only the first 10 episodes, making the revert-on-diverge check cover as
   little as 10% of training data.  The cap is now 50 episodes (or all, if
   fewer) to give a much more reliable estimate.

3. **EWC closure safety**: The ``fisher`` pytree captured inside ``_step_loss``
   is a JAX pytree reference, not a Python closure over a mutable Python value.
   This is safe because ``_step_loss`` is called inside ``eqx.filter_value_and_grad``
   which traces through it; ``fisher`` is treated as a compile-time constant.

Training protocol
-----------------
1. Compute diagonal Fisher Information Matrix on pre-training weights (up to 10 eps).
2. Initialise Optax optimizer (Adam or MESU).
3. For each of ``n_steps`` steps:
   a. Select episode by round-robin.
   b. Re-initialise carry for this episode.
   c. Compute task loss + EWC penalty.
   d. Scale gradients by PER importance weight.
   e. Zero out gradients for non-backbone parameters.
   f. Apply optimizer update.
4. Compute post-training loss on up to 50 episodes.
5. Revert to checkpoint if ``loss_after > loss_before``.
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

# Maximum episodes used for loss evaluation (revert-on-diverge check)
_MAX_LOSS_EVAL_EPS = 50


def _backbone_filter(model: HaloFEPModel):
    """Return a boolean pytree: True only for backbone leaves.

    Used to zero out gradients for all parameters outside the HALO backbone,
    so only the backbone is updated during nightly LoRA fine-tuning.
    """
    false_model    = jtu.tree_map(lambda _: False, model)
    true_backbone  = jtu.tree_map(lambda _: True, model.backbone)
    return eqx.tree_at(lambda m: m.backbone, false_model, true_backbone)


def _compute_fisher(
    model: HaloFEPModel,
    episodes: list[Episode],
    key: jnp.ndarray,
) -> Any:
    """Diagonal Fisher Information Matrix over backbone parameters.

    Approximated as the mean squared gradient of the loss w.r.t. backbone
    weights, computed over up to 10 episodes for speed.

    Each episode uses a freshly initialised carry so episodes are independent.

    Parameters
    ----------
    model    : Model to compute Fisher for.
    episodes : Training episodes (at most 10 are used).
    key      : JAX PRNGKey.

    Returns
    -------
    PyTree matching model.backbone shape with float32 diagonal Fisher values.
    """
    fisher  = jtu.tree_map(lambda x: jnp.zeros_like(x), model.backbone)
    n_eval  = min(len(episodes), 10)
    if n_eval == 0:
        return fisher  # all-zeros Fisher: no data to estimate from

    for ep in episodes[:n_eval]:
        tokens  = jnp.array(ep.tokens)
        key, sk = jax.random.split(key)

        # Fresh carry per episode — no contamination
        carry = model.init_carry(sk)
        key, sk2 = jax.random.split(key)

        grad_fn = jax.grad(
            lambda m: unified_elbo_loss(m, carry, tokens, sk2)[0]
        )
        grads = grad_fn(model)

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
    """EWC regularization: λ · Σ_i F_i · (θ_i − θ_i*)².

    Penalises deviation from the pre-training checkpoint proportionally to
    how important each parameter was (estimated via Fisher diagonal).

    Parameters
    ----------
    current_backbone    : Current backbone pytree.
    checkpoint_backbone : Pre-training checkpoint backbone pytree.
    fisher              : Diagonal Fisher pytree (same structure).
    ewc_lambda          : Penalty weight.

    Returns
    -------
    Scalar EWC penalty (float32).
    """
    diffs = jtu.tree_map(lambda c, o: c - o, current_backbone, checkpoint_backbone)
    penalties = jtu.tree_map(
        lambda f, d: jnp.sum(f * d ** 2),
        fisher,
        diffs,
    )
    return ewc_lambda * sum(jtu.tree_leaves(penalties))


class LoRATrainer:
    """Nightly backbone fine-tuning with EWC, PER, and revert-on-diverge.

    Parameters
    ----------
    cfg     : System configuration.
    n_steps : Number of gradient steps per nightly dream.
    lr      : Learning rate for the optimizer.
    """

    def __init__(
        self,
        cfg: HaloFEPConfig,
        n_steps: int = 100,
        lr: float = 1e-4,
    ) -> None:
        self.cfg     = cfg
        self.n_steps = n_steps
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
        """Fine-tune the backbone on ``episodes`` with EWC-LoRA regularization.

        Parameters
        ----------
        model       : Current model to fine-tune.
        episodes    : High-confidence episodes to train on.
        per_weights : Optional (N,) float32 IS weights from PER sampling.
                      If None, all episodes are weighted equally.

        Returns
        -------
        (model, log_dict) — model may be the original if divergence detected.

        Notes
        -----
        The ``carry`` is **re-initialised per episode** (line tagged below) to
        prevent state contamination between independent training examples.
        """
        if not episodes:
            return model, {
                "loss_before": 0.0,
                "loss_after":  0.0,
                "n_episodes":  0,
                "ewc_penalty": 0.0,
            }

        if per_weights is None:
            per_weights = np.ones(len(episodes), dtype=np.float32)
        per_weights = np.asarray(per_weights, dtype=np.float32)

        key = jax.random.PRNGKey(self.cfg.seed)

        loss_before = self._mean_loss(model, episodes, key)
        log.info(
            f"LoRA fine-tune: loss_before={loss_before:.4f}, "
            f"n_episodes={len(episodes)}"
        )

        checkpoint           = model
        checkpoint_backbone  = model.backbone  # snapshot for EWC

        # Compute Fisher on pre-update weights
        ewc_penalty_val = 0.0
        fisher = None
        if self.cfg.ewc_lambda > 0.0:
            key, fk = jax.random.split(key)
            fisher = _compute_fisher(model, episodes, fk)

        opt_state = self.opt.init(eqx.filter(model, eqx.is_array))

        for step in range(self.n_steps):
            ep_idx  = step % len(episodes)
            tokens  = jnp.array(episodes[ep_idx].tokens)
            w       = float(per_weights[ep_idx])
            key, sk = jax.random.split(key)

            # --- KEY FIX: Re-initialise carry per episode (no cross-contamination) ---
            carry = model.init_carry(sk)

            def _step_loss(m):
                """Combined task loss + EWC penalty for this episode."""
                task_loss, _ = unified_elbo_loss(m, carry, tokens, sk)
                if fisher is not None:
                    ewc = _ewc_penalty(
                        m.backbone, checkpoint_backbone, fisher, self.cfg.ewc_lambda
                    )
                    return task_loss + ewc
                return task_loss

            loss, grads = eqx.filter_value_and_grad(_step_loss)(model)

            # Scale loss gradient by PER importance weight
            grads = jtu.tree_map(lambda g: g * w, grads)

            # Zero out grads outside backbone — LoRA-style: only backbone is trained
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

            if step % 10 == 0:
                log.debug(f"  Step {step}/{self.n_steps} | loss={float(loss):.5f}")

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
            log.info(f"EWC penalty after training: {ewc_penalty_val:.4f}")

        key, ek = jax.random.split(key)
        loss_after = self._mean_loss(model, episodes, ek)
        log.info(f"LoRA fine-tune: loss_after={loss_after:.4f}")

        # Revert-on-diverge: discard new weights if loss increased
        if loss_after > loss_before:
            log.warning(
                f"Loss increased after fine-tuning "
                f"({loss_after:.6f} > {loss_before:.6f}) — reverting to checkpoint."
            )
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
        episodes: list[Episode],
        key: jnp.ndarray,
    ) -> float:
        """Compute mean ELBO loss over up to ``_MAX_LOSS_EVAL_EPS`` episodes.

        Each episode uses a freshly initialised carry for consistent comparison
        across calls (before/after training).

        Parameters
        ----------
        model    : Model to evaluate.
        episodes : Training episodes (capped at _MAX_LOSS_EVAL_EPS).
        key      : JAX PRNGKey.

        Returns
        -------
        Mean scalar loss as a Python float.
        """
        losses = []
        eval_eps = episodes[:_MAX_LOSS_EVAL_EPS]
        for ep in eval_eps:
            tokens   = jnp.array(ep.tokens)
            key, sk  = jax.random.split(key)
            carry    = model.init_carry(sk)          # fresh carry per episode
            key, sk2 = jax.random.split(key)
            loss, _  = unified_elbo_loss(model, carry, tokens, sk2)
            losses.append(float(loss))
        return float(np.mean(losses)) if losses else 0.0
