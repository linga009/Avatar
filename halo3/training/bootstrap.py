"""Bootstrap pre-training for HoloBiont 3.0."""
from __future__ import annotations
import logging
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from halo3.config import Halo3Config
from halo3.model import Halo3Model
from halo3.loss import halo3_loss

log = logging.getLogger(__name__)


def save_checkpoint(model, path, keep_versions=3):
    """Save model checkpoint with rotation — keeps last `keep_versions` backups."""
    directory = os.path.dirname(path) if os.path.dirname(path) else "."
    os.makedirs(directory, exist_ok=True)
    eqx_path = path + ".eqx"

    # Rotate existing backups before overwriting
    if keep_versions > 0 and os.path.exists(eqx_path):
        import glob as _glob
        base = path
        # Delete oldest if we're at the limit
        existing = sorted(_glob.glob(base + ".bak*.eqx"))
        while len(existing) >= keep_versions:
            os.remove(existing[0])
            existing.pop(0)
        # Shift current to next backup slot
        bak_num = len(existing) + 1
        import shutil
        shutil.copy2(eqx_path, f"{base}.bak{bak_num}.eqx")

    eqx.tree_serialise_leaves(eqx_path, model)


def load_checkpoint(cfg, path):
    template = Halo3Model(cfg, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path + ".eqx", template)


@eqx.filter_jit
def _train_step(model, opt_state, carry, tokens, key, opt):
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


def run_bootstrap(cfg, n_steps=5000, checkpoint_dir="data/checkpoints/halo3", seed=42):
    key = jax.random.PRNGKey(seed)
    k1, _ = jax.random.split(key)
    model = Halo3Model(cfg, k1)
    carry = model.init_carry(key)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=cfg.lr,
        warmup_steps=500, decay_steps=n_steps, end_value=cfg.lr * 0.1,
    )
    opt = optax.adafactor(learning_rate=schedule)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    log.info(f"JIT compiling training step (first call will be slow)...")

    for step in range(n_steps):
        key, sk = jax.random.split(key)
        tokens = jax.random.normal(sk, (cfg.n_tokens, cfg.d_model))

        model, opt_state, loss = _train_step(
            model, opt_state, carry, tokens, sk, opt
        )

        if step % 500 == 0:
            log.info(f"Step {step}/{n_steps} | loss={float(loss):.4f}")

    save_checkpoint(model, checkpoint_dir)
    return model
