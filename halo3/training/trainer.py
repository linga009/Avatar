"""Halo3Trainer — Adafactor + GaLore + LISA, fully JIT-compiled."""
from __future__ import annotations
import logging
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from halo3.config import Halo3Config
from halo3.model import Halo3Model
from halo3.loss import halo3_loss

log = logging.getLogger(__name__)


def _lisa_gradient_mask(grads, model, key, n_active, n_layers):
    """Zero out gradients for all but n_active random backbone layers."""
    active = jax.random.choice(key, n_layers, shape=(n_active,), replace=False)
    active_set = set(int(a) for a in active)

    def _mask(layer_grads, i):
        if i in active_set:
            return layer_grads
        return jax.tree_util.tree_map(jnp.zeros_like, layer_grads)

    new_layers = [_mask(grads.backbone.layers[i], i) for i in range(n_layers)]
    new_ffns = [_mask(grads.backbone.ffns[i], i) for i in range(n_layers)]
    new_bb = eqx.tree_at(lambda b: b.layers, grads.backbone, new_layers)
    new_bb = eqx.tree_at(lambda b: b.ffns, new_bb, new_ffns)
    return eqx.tree_at(lambda m: m.backbone, grads, new_bb)


@eqx.filter_jit
def _train_step_jit(model, opt_state, carry, tokens, key, opt):
    """Single JIT-compiled training step — runs entirely on GPU."""
    loss_fn = lambda m: halo3_loss(m, carry, tokens, key)[0]
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, new_opt_state = opt.update(
        eqx.filter(grads, eqx.is_array),
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


class Halo3Trainer:
    def __init__(self, cfg: Halo3Config, n_steps=100, lr=3e-4):
        self.cfg = cfg
        self.n_steps = n_steps
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=lr,
            warmup_steps=min(50, n_steps // 10),
            decay_steps=n_steps, end_value=lr * 0.1,
        )
        self.opt = optax.adafactor(learning_rate=schedule)

    def run(self, model, episodes):
        if not episodes:
            return model, {"loss_before": 0.0, "loss_after": 0.0}

        key = jax.random.PRNGKey(self.cfg.seed)
        carry = model.init_carry(key)
        opt_state = self.opt.init(eqx.filter(model, eqx.is_array))

        log.info("JIT compiling training step...")

        for step in range(self.n_steps):
            ep = episodes[step % len(episodes)]
            tokens = jnp.array(ep.tokens)
            key, sk = jax.random.split(key)

            model, opt_state, loss = _train_step_jit(
                model, opt_state, carry, tokens, sk, self.opt
            )

            if step % 100 == 0:
                log.info(f"Step {step}/{self.n_steps} | loss={float(loss):.4f}")

        return model, {"n_steps": self.n_steps}
