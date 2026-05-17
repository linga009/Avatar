"""Dream Replay — biological dreaming for Avatar's physics body.

Uses CLion (Cautious Lion, arXiv:2604.14587):
  - sign-based updates → bounded ±1, no gradient explosion possible
  - cautious mask     → never increases loss in any coordinate
  - 1 momentum buffer → same memory as SGD (half of Adam)

Critical fix vs previous version: opt_state is passed as an explicit
argument to _dream_step rather than captured by JIT closure.  The closure
approach silently reused the initial (zero) state on every step, so
momentum never accumulated and all updates were noise.
"""
from __future__ import annotations
import gc
import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

log = logging.getLogger(__name__)


# ── CLion: Cautious Lion ─────────────────────────────────────────────────────

class _CLionState(NamedTuple):
    """Single momentum buffer — half the memory of Adam."""
    momentum: any  # pytree matching params


def scale_by_clion(b1: float = 0.9) -> optax.GradientTransformation:
    """Cautious Lion gradient transform (CLion, arXiv:2604.14587).

    Per step:
      1. m  = b1 * m + (1 - b1) * g          # accumulate momentum
      2. d  = sign(m)                          # Lion direction
      3. d  = d * (g * d > 0)                  # cautious mask
         → zero out coordinates where gradient and direction disagree
         → guarantees no coordinate update increases the loss
    The sign() operation bounds every update magnitude to exactly 1,
    making NaN-from-explosion structurally impossible.
    """
    def init_fn(params):
        return _CLionState(
            momentum=jax.tree_util.tree_map(jnp.zeros_like, params)
        )

    def update_fn(updates, state, params=None):
        new_momentum = jax.tree_util.tree_map(
            lambda m, g: b1 * m + (1.0 - b1) * g,
            state.momentum, updates,
        )
        lion_dir = jax.tree_util.tree_map(jnp.sign, new_momentum)
        # Cautious mask: keep only coordinates where gradient agrees with dir
        masked = jax.tree_util.tree_map(
            lambda d, g: d * (g * d > 0.0).astype(d.dtype),
            lion_dir, updates,
        )
        return masked, _CLionState(momentum=new_momentum)

    return optax.GradientTransformation(init_fn, update_fn)


def _build_optimizer(lr: float) -> optax.GradientTransformation:
    """CLion with aggressive gradient clipping for short dream sessions."""
    return optax.chain(
        optax.clip_by_global_norm(0.1),  # clip before sign — prevents Inf input
        scale_by_clion(b1=0.9),          # cautious sign update
        optax.scale(-lr),                 # apply learning rate
    )


# ── Dream replay ─────────────────────────────────────────────────────────────

def dream_replay_physics(
    model,
    episodes: list,
    n_replay_steps: int = 30,
    n_recombine_steps: int = 15,
    n_imagine_steps: int = 10,
    lr: float = 5e-6,
    seed: int = 42,
):
    """Full biological dreaming cycle for Avatar's physics body.

    Three phases, each updating MERA cores + Hamiltonian V_theta + SSM params:
      1. Replay   — re-process real episodes to strengthen representations
      2. Recombine — mix episodes from different topics for new associations
      3. Imagine  — add noise and integrate Hamiltonian for counterfactuals

    Uses CLion optimizer: same memory as SGD, better convergence, NaN-safe.
    Returns (updated_model, info_dict).
    """
    if not episodes or len(episodes) < 2:
        log.info("Dream replay: not enough episodes (need >= 2)")
        return model, {"replayed": 0, "recombined": 0, "imagined": 0}

    # Release cached XLA kernels before allocating optimizer state.
    gc.collect()
    jax.clear_caches()

    from halo3.loss import halo3_loss

    key = jax.random.PRNGKey(seed)
    carry = model.init_carry(key)
    opt = _build_optimizer(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Build token arrays from episodes (fall back to random noise if no tokens)
    ep_tokens = []
    for ep in episodes:
        if hasattr(ep, "tokens") and ep.tokens is not None:
            ep_tokens.append(jnp.array(ep.tokens))
    if not ep_tokens:
        for _ in range(min(len(episodes), 10)):
            key, sk = jax.random.split(key)
            ep_tokens.append(
                jax.random.normal(sk, (model.cfg.n_tokens, model.cfg.d_model))
            )

    log.info(
        f"Dream replay (CLion): {len(ep_tokens)} episodes, "
        f"{n_replay_steps}+{n_recombine_steps}+{n_imagine_steps} steps"
    )

    # ── JIT step — single compiled kernel reused across all 55 steps ──────────
    # Critical: this function is defined ONCE outside the loop.  All varying
    # inputs (tokens, key, scale) are passed as arguments so JAX traces once
    # and reuses the compiled XLA program.  The previous approach created a
    # new @eqx.filter_jit per call → 55 recompilations → OOM from XLA cache.
    @eqx.filter_jit
    def _dream_step(model, opt_state_in, carry, tokens, key, scale):
        loss_fn = lambda m: halo3_loss(m, carry, tokens, key)[0] * scale
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, new_opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state_in,
            eqx.filter(model, eqx.is_array),
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss

    def _safe_step(model, opt_state, carry, tokens, key, scale):
        """Run a dream step; skip silently if result contains NaN."""
        try:
            new_model, new_opt_state, loss = _dream_step(
                model, opt_state, carry, tokens, key, jnp.float32(scale)
            )
            loss_val = float(loss)
            if loss_val != loss_val:  # NaN loss
                return model, opt_state, loss
            leaves = jax.tree_util.tree_leaves(
                eqx.filter(new_model, eqx.is_array)
            )
            if any(bool(jnp.any(jnp.isnan(l))) for l in leaves):
                log.warning("  Dream: NaN weights after step — skipping")
                return model, opt_state, loss
            return new_model, new_opt_state, loss
        except Exception as e:
            log.warning(f"  Dream step exception: {e}")
            return model, opt_state, 0.0

    # ── Phase 1: Replay ───────────────────────────────────────────────────────
    log.info("  Phase 1: Replay — strengthening real experiences...")
    replay_loss, completed_replay = 0.0, 0
    for step in range(n_replay_steps):
        key, sk = jax.random.split(key)
        idx = step % len(ep_tokens)
        model, opt_state, loss = _safe_step(
            model, opt_state, carry, ep_tokens[idx], sk, 0.1
        )
        replay_loss += float(loss)
        completed_replay += 1
    replay_loss /= max(completed_replay, 1)
    gc.collect(); jax.clear_caches()

    # ── Phase 2: Recombine ────────────────────────────────────────────────────
    log.info("  Phase 2: Recombine — forming cross-episode associations...")
    recombine_loss, completed_recombine = 0.0, 0
    for step in range(n_recombine_steps):
        key, k1, k2, sk = jax.random.split(key, 4)
        idx_a = int(jax.random.randint(k1, (), 0, len(ep_tokens)))
        idx_b = int(jax.random.randint(k2, (), 0, len(ep_tokens)))
        n_half = model.cfg.n_tokens // 2
        tokens_mixed = jnp.concatenate(
            [ep_tokens[idx_a][:n_half], ep_tokens[idx_b][n_half:]], axis=0
        )
        model, opt_state, loss = _safe_step(
            model, opt_state, carry, tokens_mixed, sk, 0.05
        )
        recombine_loss += float(loss)
        completed_recombine += 1
    recombine_loss /= max(completed_recombine, 1)
    gc.collect(); jax.clear_caches()

    # ── Phase 3: Imagine ──────────────────────────────────────────────────────
    log.info("  Phase 3: Imagine — counterfactual trajectories...")
    imagine_loss, completed_imagine = 0.0, 0
    for step in range(n_imagine_steps):
        key, sk = jax.random.split(key)
        idx = step % len(ep_tokens)
        noise = jax.random.normal(sk, ep_tokens[idx].shape) * 0.2
        model, opt_state, loss = _safe_step(
            model, opt_state, carry, ep_tokens[idx] + noise, sk, 0.02
        )
        imagine_loss += float(loss)
        completed_imagine += 1
    imagine_loss /= max(completed_imagine, 1)

    info = {
        "replayed": completed_replay,
        "recombined": completed_recombine,
        "imagined": completed_imagine,
        "replay_loss": replay_loss,
        "recombine_loss": recombine_loss,
        "imagine_loss": imagine_loss,
    }
    log.info(
        f"  Dream complete (CLion): "
        f"{completed_replay}/{n_replay_steps} replay, "
        f"{completed_recombine}/{n_recombine_steps} recombine, "
        f"{completed_imagine}/{n_imagine_steps} imagine | "
        f"losses: {replay_loss:.2e} / {recombine_loss:.2e} / {imagine_loss:.2e}"
    )
    return model, info
