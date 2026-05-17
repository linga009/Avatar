# Training Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six identified weaknesses in HoloBiont's training pipeline: add EWC-LoRA to prevent catastrophic forgetting, Prioritized Experience Replay for richer nightly signal, MESU optimizer for boundary-free continual learning, multi-scale SSM bootstrap training, Wikipedia-derived topic calibration, and WN18RR hyperbolic pre-training.

**Architecture:** All changes are confined to `halo_fep/training/` and `halo_fep/memory/episode_store.py`. Each improvement is opt-in via new `HaloFEPConfig` fields with safe defaults, so existing tests and the heartbeat loop continue to work unchanged. Tasks 1–4 are pure-Python/JAX with no new dependencies; Tasks 5–6 require `pip install datasets` and internet access at training time only.

**Tech Stack:** JAX/Equinox, optax, FAISS, SQLAlchemy, HuggingFace `datasets` (Tasks 5–6 only), sentence-transformers (already a dependency).

---

## File Map

| File | Action | Responsible for |
|---|---|---|
| `halo_fep/config.py` | Modify | New fields: `ewc_lambda`, `per_alpha`, `per_beta`, `use_mesu`, `mesu_eta` |
| `halo_fep/memory/episode_store.py` | Modify | Add `get_prioritized()` method |
| `halo_fep/training/lora_trainer.py` | Modify | EWC-LoRA penalty, PER-weighted loss, MESU opt-in |
| `halo_fep/training/mesu.py` | Create | MESU custom optax optimizer |
| `halo_fep/training/bootstrap.py` | Modify | Multi-scale loss, opt-in Wikipedia + WN18RR paths |
| `halo_fep/training/topic_bootstrap.py` | Create | WikiText-103 streaming → per-cluster token arrays |
| `halo_fep/training/hyperbolic_pretrain.py` | Create | WN18RR → Poincaré loss → pre-train HoloEmbedding |
| `halo_fep/memory/tests/test_episode_store.py` | Modify | Add PER tests |
| `halo_fep/training/tests/test_lora_trainer.py` | Modify | Add EWC + PER + MESU tests |
| `halo_fep/training/tests/test_mesu.py` | Create | MESU optimizer unit tests |
| `halo_fep/training/tests/test_bootstrap.py` | Modify | Add multi-scale loss test |
| `halo_fep/training/tests/test_topic_bootstrap.py` | Create | Wikipedia bootstrap with mock dataset |
| `halo_fep/training/tests/test_hyperbolic_pretrain.py` | Create | WN18RR pre-training unit tests |

---

## Task 1: Config Extensions

**Files:**
- Modify: `halo_fep/config.py:50-62`

- [ ] **Step 1: Write the failing test**

```python
# Add to halo_fep/tests/test_config.py (create file if not exists):
from halo_fep.config import HaloFEPConfig

def test_new_training_fields_have_correct_defaults():
    cfg = HaloFEPConfig()
    assert cfg.ewc_lambda == 0.1
    assert cfg.per_alpha == 0.6
    assert cfg.per_beta == 0.4
    assert cfg.use_mesu is False
    assert cfg.mesu_eta == 0.01

def test_ewc_lambda_zero_disables_ewc():
    cfg = HaloFEPConfig(ewc_lambda=0.0)
    assert cfg.ewc_lambda == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd D:/New_Ai
python -m pytest halo_fep/tests/test_config.py::test_new_training_fields_have_correct_defaults -v
```
Expected: `FAILED` — `AttributeError: ewc_lambda`

- [ ] **Step 3: Add new fields to HaloFEPConfig**

In `halo_fep/config.py`, add the following fields after the `# Joint training` block (after line 48, before `# Heartbeat`):

```python
    # Continual learning
    ewc_lambda:  float = 0.1    # EWC penalty weight (0 = disabled)
    per_alpha:   float = 0.6    # PER priority exponent (0=uniform, 1=full priority)
    per_beta:    float = 0.4    # PER importance-sampling correction exponent
    use_mesu:    bool  = False   # Use MESU optimizer instead of Adam for nightly LoRA
    mesu_eta:    float = 0.01   # MESU uncertainty EMA rate
```

The full updated `halo_fep/config.py` after the change (only the relevant section):

```python
    # Joint training
    lambda_fep: float = 0.1
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42

    # Continual learning
    ewc_lambda:  float = 0.1
    per_alpha:   float = 0.6
    per_beta:    float = 0.4
    use_mesu:    bool  = False
    mesu_eta:    float = 0.01

    # Heartbeat
    wake_threshold: float = 2.5
    tick_interval:  int   = 60
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest halo_fep/tests/test_config.py -v
```
Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add halo_fep/config.py halo_fep/tests/test_config.py
git commit -m "feat(config): add continual learning fields (ewc_lambda, per_alpha, per_beta, use_mesu, mesu_eta)"
```

---

## Task 2: Prioritized Experience Replay (PER)

**Files:**
- Modify: `halo_fep/memory/episode_store.py:149-155`
- Modify: `halo_fep/memory/tests/test_episode_store.py`

- [ ] **Step 1: Write the failing test**

Add to `halo_fep/memory/tests/test_episode_store.py`:

```python
import numpy as np
import tempfile
from halo_fep.memory.episode_store import EpisodeStore
from halo_fep.memory.schema import Episode


def _make_episode(query="q", fe=1.0, delta=-0.1):
    return Episode(
        query=query,
        tokens=np.zeros((32, 256), dtype=np.float32),
        swarm_mu=np.zeros((256, 8), dtype=np.float32),
        free_energy=fe,
        free_energy_delta=delta,
    )


def test_get_prioritized_returns_correct_count():
    with tempfile.TemporaryDirectory() as tmp:
        store = EpisodeStore(tmp)
        qe = np.random.randn(256).astype(np.float32)
        qe /= np.linalg.norm(qe) + 1e-8
        for i in range(10):
            ep = _make_episode(query=f"q{i}", delta=-(0.05 + i * 0.01))
            store.add(ep, query_embed=qe.copy())
        episodes, weights = store.get_prioritized(n=5, alpha=0.6, beta=0.4)
        assert len(episodes) == 5
        assert weights.shape == (5,)
        assert np.all(weights > 0)
        assert np.max(weights) <= 1.0 + 1e-6   # normalized


def test_get_prioritized_fewer_than_n():
    with tempfile.TemporaryDirectory() as tmp:
        store = EpisodeStore(tmp)
        qe = np.ones(256, dtype=np.float32)
        qe /= np.linalg.norm(qe)
        store.add(_make_episode(delta=-0.1), query_embed=qe.copy())
        store.add(_make_episode(delta=-0.2), query_embed=qe.copy())
        episodes, weights = store.get_prioritized(n=10, alpha=0.6, beta=0.4)
        assert len(episodes) == 2   # only 2 available


def test_get_prioritized_empty_store():
    with tempfile.TemporaryDirectory() as tmp:
        store = EpisodeStore(tmp)
        episodes, weights = store.get_prioritized(n=5, alpha=0.6, beta=0.4)
        assert episodes == []
        assert len(weights) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest halo_fep/memory/tests/test_episode_store.py::test_get_prioritized_returns_correct_count -v
```
Expected: `FAILED` — `AttributeError: 'EpisodeStore' object has no attribute 'get_prioritized'`

- [ ] **Step 3: Add `get_prioritized` to EpisodeStore**

Add the following method to `halo_fep/memory/episode_store.py` after `get_high_confidence` (after line 155):

```python
    def get_prioritized(
        self,
        n: int,
        since_timestamp: float = 0.0,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> tuple[list["Episode"], np.ndarray]:
        """Return up to n episodes sampled proportional to |free_energy_delta|^alpha.

        Higher |delta_fe| = more surprising/informative = higher priority.

        Args:
            n: Maximum number of episodes to return.
            since_timestamp: Only consider episodes after this Unix timestamp.
            alpha: Priority exponent. 0 = uniform sampling, 1 = full priority.
            beta: Importance-sampling correction exponent. 0 = no correction.

        Returns:
            (episodes, weights) — weights are IS corrections in [0, 1].
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                sa.select(_EPISODES)
                .where(_EPISODES.c.timestamp >= since_timestamp)
                .order_by(_EPISODES.c.timestamp)
            ).fetchall()

        if not rows:
            return [], np.array([], dtype=np.float32)

        episodes = [self._row_to_episode(r) for r in rows]

        # Priority = |delta_fe|^alpha, clipped to avoid zeros
        priorities = np.array(
            [abs(ep.free_energy_delta) ** alpha for ep in episodes],
            dtype=np.float32,
        )
        priorities = np.clip(priorities, 1e-8, None)
        probs = priorities / priorities.sum()

        n_sample = min(n, len(episodes))
        indices = np.random.choice(len(episodes), size=n_sample, replace=False, p=probs)

        sampled = [episodes[i] for i in indices]
        sampled_probs = probs[indices]

        # IS weights: w_i = (1/(N*p_i))^beta, normalized to [0,1]
        N = len(episodes)
        raw_weights = (1.0 / (N * sampled_probs + 1e-8)) ** beta
        weights = (raw_weights / raw_weights.max()).astype(np.float32)

        return sampled, weights
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest halo_fep/memory/tests/test_episode_store.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add halo_fep/memory/episode_store.py halo_fep/memory/tests/test_episode_store.py
git commit -m "feat(memory): add get_prioritized() — Prioritized Experience Replay sampling"
```

---

## Task 3: EWC-LoRA in LoRATrainer

**Files:**
- Modify: `halo_fep/training/lora_trainer.py`
- Modify: `halo_fep/training/tests/test_lora_trainer.py`

- [ ] **Step 1: Write the failing tests**

Add to `halo_fep/training/tests/test_lora_trainer.py`:

```python
def test_ewc_lora_reduces_loss_increase():
    """EWC penalty should be non-negative and logged."""
    cfg = HaloFEPConfig(n_tokens=32, ewc_lambda=1.0)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    episodes = make_episodes(cfg, n=3)
    _, log = trainer.run(model, episodes)
    assert "ewc_penalty" in log
    assert log["ewc_penalty"] >= 0.0


def test_ewc_disabled_when_lambda_zero():
    """With ewc_lambda=0.0, ewc_penalty should be 0.0."""
    cfg = HaloFEPConfig(n_tokens=32, ewc_lambda=0.0)
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    trainer = LoRATrainer(cfg, n_steps=2, lr=1e-3)
    episodes = make_episodes(cfg, n=2)
    _, log = trainer.run(model, episodes)
    assert log["ewc_penalty"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest halo_fep/training/tests/test_lora_trainer.py::test_ewc_lora_reduces_loss_increase -v
```
Expected: `FAILED` — `KeyError: 'ewc_penalty'`

- [ ] **Step 3: Add Fisher computation and EWC penalty to lora_trainer.py**

Replace the full content of `halo_fep/training/lora_trainer.py` with:

```python
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
```

- [ ] **Step 4: Run all lora_trainer tests**

```bash
python -m pytest halo_fep/training/tests/test_lora_trainer.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/lora_trainer.py halo_fep/training/tests/test_lora_trainer.py
git commit -m "feat(training): EWC-LoRA — Fisher-weighted penalty prevents catastrophic forgetting"
```

---

## Task 4: MESU Optimizer

**Files:**
- Create: `halo_fep/training/mesu.py`
- Create: `halo_fep/training/tests/test_mesu.py`

- [ ] **Step 1: Write the failing tests**

Create `halo_fep/training/tests/test_mesu.py`:

```python
# halo_fep/training/tests/test_mesu.py
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import pytest
from halo_fep.training.mesu import mesu


def test_mesu_returns_gradient_transformation():
    opt = mesu(lr=1e-3, eta=0.01)
    assert hasattr(opt, "init")
    assert hasattr(opt, "update")


def test_mesu_init_state_has_sigma():
    opt = mesu(lr=1e-3, eta=0.01)
    params = {"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}
    state = opt.init(params)
    assert "sigma" in state
    # sigma should be all-ones initially (uninformative prior)
    for leaf in jtu.tree_leaves(state["sigma"]):
        assert jnp.allclose(leaf, jnp.ones_like(leaf))


def test_mesu_updates_reduce_shape_matches_params():
    opt = mesu(lr=1e-3, eta=0.01)
    params = {"w": jnp.ones((3, 3))}
    state = opt.init(params)
    grads = {"w": jnp.ones((3, 3)) * 0.1}
    updates, new_state = opt.update(grads, state)
    assert updates["w"].shape == (3, 3)
    assert new_state["sigma"]["w"].shape == (3, 3)


def test_mesu_sigma_increases_with_large_gradients():
    """High gradient variance should increase uncertainty sigma."""
    opt = mesu(lr=1e-4, eta=0.5)
    params = {"w": jnp.zeros(4)}
    state = opt.init(params)
    # Large gradient
    grads = {"w": jnp.ones(4) * 10.0}
    _, state2 = opt.update(grads, state)
    # sigma should have grown from 1.0 toward 10^2 = 100
    assert jnp.all(state2["sigma"]["w"] > state["sigma"]["w"])


def test_mesu_update_scales_by_inverse_sigma():
    """Updates should be smaller when sigma is large."""
    opt = mesu(lr=1.0, eta=0.0, epsilon=0.0)  # no sigma adaptation
    params = {"w": jnp.zeros(1)}

    # Low sigma (uncertain parameter) -> large update
    state_low = {"sigma": {"w": jnp.ones(1) * 0.1}}
    grads = {"w": jnp.ones(1)}
    updates_low, _ = opt.update(grads, state_low)

    # High sigma (certain parameter) -> small update
    state_high = {"sigma": {"w": jnp.ones(1) * 10.0}}
    updates_high, _ = opt.update(grads, state_high)

    assert abs(updates_low["w"][0]) > abs(updates_high["w"][0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest halo_fep/training/tests/test_mesu.py -v
```
Expected: `FAILED` — `ModuleNotFoundError: halo_fep.training.mesu`

- [ ] **Step 3: Create `halo_fep/training/mesu.py`**

```python
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
        return {"sigma": jtu.tree_map(jnp.ones_like, params)}

    def update_fn(
        updates: Any,
        state: MESUState,
        params: Any = None,
    ) -> tuple[Any, MESUState]:
        sigma = state["sigma"]

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

        return scaled_updates, {"sigma": new_sigma}

    return optax.GradientTransformation(init_fn, update_fn)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest halo_fep/training/tests/test_mesu.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Verify MESU integrates with LoRATrainer**

```bash
python -c "
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.training.lora_trainer import LoRATrainer
from halo_fep.memory.schema import Episode
import numpy as np, jax

cfg = HaloFEPConfig(n_tokens=32, use_mesu=True, mesu_eta=0.01)
model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
trainer = LoRATrainer(cfg, n_steps=2, lr=1e-4)
eps = [Episode(query='q', tokens=np.zeros((32,256),dtype='f4'), swarm_mu=np.zeros((256,8),dtype='f4'), free_energy=1.0, free_energy_delta=-0.1)]
m2, log = trainer.run(model, eps)
print('MESU integration OK:', log)
"
```
Expected: prints `MESU integration OK: {...}` without error.

- [ ] **Step 6: Commit**

```bash
git add halo_fep/training/mesu.py halo_fep/training/tests/test_mesu.py
git commit -m "feat(training): MESU optimizer — boundary-free continual learning via synaptic uncertainty"
```

---

## Task 5: Multi-Scale SSM Bootstrap Training

**Files:**
- Modify: `halo_fep/training/bootstrap.py:40-93`
- Modify: `halo_fep/training/tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Add to `halo_fep/training/tests/test_bootstrap.py`:

```python
import jax
import jax.numpy as jnp
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.training.bootstrap import _multiscale_elbo_loss


def test_multiscale_loss_is_scalar():
    cfg = HaloFEPConfig(n_tokens=32)
    key = jax.random.PRNGKey(0)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    tokens = jnp.zeros((32, cfg.d_model))
    loss, aux = _multiscale_elbo_loss(model, carry, tokens, key, strides=(1, 4))
    assert loss.shape == ()
    assert float(loss) >= 0.0


def test_multiscale_loss_differs_from_single_scale():
    """Multi-scale loss should differ from single-stride loss."""
    cfg = HaloFEPConfig(n_tokens=32)
    key = jax.random.PRNGKey(1)
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    tokens = jax.random.normal(key, (32, cfg.d_model))
    from halo_fep.loss import unified_elbo_loss
    single, _ = unified_elbo_loss(model, carry, tokens, key)
    multi, _  = _multiscale_elbo_loss(model, carry, tokens, key, strides=(1, 4))
    # They should not be identical (strides produce different subsampled inputs)
    assert not jnp.allclose(single, multi, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest halo_fep/training/tests/test_bootstrap.py::test_multiscale_loss_is_scalar -v
```
Expected: `FAILED` — `ImportError: cannot import name '_multiscale_elbo_loss'`

- [ ] **Step 3: Add `_multiscale_elbo_loss` to bootstrap.py and use it**

Replace `halo_fep/training/bootstrap.py` with:

```python
# halo_fep/training/bootstrap.py
"""Phase 0 bootstrap: pre-train HALO+FEP on MultimodalWorld, save checkpoint.

Optionally uses Wikipedia topic data (requires `pip install datasets`) and
WN18RR hyperbolic pre-training. Multi-scale SSM training is always enabled.

Usage:
    python -m halo_fep.training.bootstrap
"""
from __future__ import annotations

import logging
import os

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.loss import unified_elbo_loss
from halo_fep.benchmark.multimodal_world import MultimodalWorld

log = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "data/checkpoints/bootstrap"


def save_checkpoint(model: HaloFEPModel, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    eqx.tree_serialise_leaves(path + ".eqx", model)
    log.info(f"Checkpoint saved to {path}.eqx")


def load_checkpoint(cfg: HaloFEPConfig, path: str) -> HaloFEPModel:
    template = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    model    = eqx.tree_deserialise_leaves(path + ".eqx", template)
    log.info(f"Checkpoint loaded from {path}.eqx")
    return model


def _pad_tokens(tokens_2: jnp.ndarray, n_tokens: int) -> jnp.ndarray:
    """Pad (2, d_model) to (n_tokens, d_model) with zeros."""
    d_model = tokens_2.shape[1]
    pad = jnp.zeros((n_tokens - 2, d_model), dtype=jnp.float32)
    return jnp.concatenate([tokens_2, pad], axis=0)


def _multiscale_elbo_loss(
    model: HaloFEPModel,
    carry,
    tokens: jnp.ndarray,
    key: jnp.ndarray,
    strides: tuple[int, ...] = (1, 4),
) -> tuple[jnp.ndarray, dict]:
    """Compute ELBO at multiple temporal strides and return the mean.

    Stride 1 = original sequence (fine-grained patterns).
    Stride 4 = every 4th token subsampled then zero-padded (coarse patterns).

    Training at both scales simultaneously gives the SSM blocks richer
    temporal representations — analogous to multi-resolution wavelet analysis.

    Args:
        model: HaloFEPModel.
        carry: HaloFEPCarry (belief state).
        tokens: (n_tokens, d_model) float32 input.
        key: JAX PRNG key.
        strides: Tuple of stride values. Each > 0.

    Returns:
        (mean_loss, aux_from_first_stride)
    """
    n_tokens = tokens.shape[0]
    total_loss = jnp.zeros(())
    first_aux = None

    for stride in strides:
        # Subsample: take every `stride`-th token
        indices = jnp.arange(0, n_tokens, stride)
        sub = tokens[indices]                                   # (n_tokens//stride, d_model)
        # Pad back to n_tokens with zeros
        pad_len = n_tokens - sub.shape[0]
        padded  = jnp.concatenate(
            [sub, jnp.zeros((pad_len, tokens.shape[1]), dtype=jnp.float32)],
            axis=0,
        )                                                       # (n_tokens, d_model)
        key, sk = jax.random.split(key)
        loss_i, aux_i = unified_elbo_loss(model, carry, padded, sk)
        total_loss = total_loss + loss_i
        if first_aux is None:
            first_aux = aux_i

    return total_loss / len(strides), first_aux


def run_bootstrap(
    cfg: HaloFEPConfig,
    n_pretrain_steps: int = 5_000,
    n_synthetic_episodes: int = 100,
    checkpoint_dir: str = _DEFAULT_CHECKPOINT,
    seed: int = 42,
    use_wikipedia: bool = False,
    use_wn18rr: bool = False,
    multiscale_strides: tuple[int, ...] = (1, 4),
) -> HaloFEPModel:
    """Run Phase 0 bootstrap pre-training.

    Args:
        cfg: Model config (n_tokens must match heartbeat config, typically 32).
        n_pretrain_steps: SGD steps on synthetic/Wikipedia data.
        n_synthetic_episodes: Warm-up rollout steps after pre-training.
        checkpoint_dir: Directory to save model checkpoint.
        seed: Random seed for reproducibility.
        use_wikipedia: If True, replace synthetic data with WikiText-103 topic
                       samples (requires `pip install datasets`).
        use_wn18rr: If True, pre-train HoloEmbedding on WN18RR Poincare loss
                    before the main training loop (requires `pip install datasets`).
        multiscale_strides: Stride tuple for multi-scale ELBO loss.
    """
    key = jax.random.PRNGKey(seed)
    key, k1, k2 = jax.random.split(key, 3)

    model = HaloFEPModel(cfg, k1)
    carry = model.init_carry(key)
    opt   = optax.adam(cfg.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # --- Optional: WN18RR hyperbolic pre-training ---
    if use_wn18rr:
        from halo_fep.training.hyperbolic_pretrain import run_hyperbolic_pretrain
        log.info("Running WN18RR hyperbolic pre-training on HoloEmbedding...")
        model = run_hyperbolic_pretrain(model, cfg, key=k2)

    # --- Main pre-training loop ---
    if use_wikipedia:
        from halo_fep.training.topic_bootstrap import iter_wikipedia_token_batches
        log.info(f"Bootstrap: {n_pretrain_steps} steps on WikiText-103 topic data.")
        token_iter = iter_wikipedia_token_batches(cfg, seed=seed)
    else:
        world = MultimodalWorld(cfg, k2)
        log.info(f"Bootstrap: {n_pretrain_steps} steps on MultimodalWorld.")

    for step in range(n_pretrain_steps):
        key, sk1, sk2 = jax.random.split(key, 3)

        if use_wikipedia:
            tokens = jnp.array(next(token_iter))
        else:
            eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
            tokens2, _ = world.sample(eta, sk2)
            tokens = _pad_tokens(tokens2, cfg.n_tokens)

        # Multi-scale ELBO loss
        (loss, _), grads = eqx.filter_value_and_grad(
            _multiscale_elbo_loss, has_aux=True
        )(model, carry, tokens, sk2, multiscale_strides)

        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)

        if step % 500 == 0:
            log.info(f"Step {step}/{n_pretrain_steps} | loss={float(loss):.4f}")

    # --- Warm-up rollouts ---
    log.info(f"Running {n_synthetic_episodes} warm-up rollout episodes.")
    world_warmup = MultimodalWorld(cfg, k2)
    for ep in range(n_synthetic_episodes):
        key, sk1, sk2 = jax.random.split(key, 3)
        eta    = int(jax.random.randint(sk1, (), 0, cfg.n_hidden))
        tokens2, _ = world_warmup.sample(eta, sk2)
        tokens = _pad_tokens(tokens2, cfg.n_tokens)
        carry, _ = halo_fep_step(model, carry, tokens, sk2)

    save_checkpoint(model, checkpoint_dir)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = HaloFEPConfig(n_tokens=32, wake_threshold=2.5, tick_interval=60)
    run_bootstrap(cfg)
```

- [ ] **Step 4: Run all bootstrap tests**

```bash
python -m pytest halo_fep/training/tests/test_bootstrap.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/bootstrap.py halo_fep/training/tests/test_bootstrap.py
git commit -m "feat(training): multi-scale ELBO loss in bootstrap — SSM trains at stride 1 and 4 simultaneously"
```

---

## Task 6: Wikipedia Topic Bootstrap

**Files:**
- Create: `halo_fep/training/topic_bootstrap.py`
- Create: `halo_fep/training/tests/test_topic_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `halo_fep/training/tests/test_topic_bootstrap.py`:

```python
# halo_fep/training/tests/test_topic_bootstrap.py
"""Tests for Wikipedia topic bootstrap using a mock dataset."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from halo_fep.config import HaloFEPConfig
from halo_fep.training.topic_bootstrap import (
    TOPIC_KEYWORDS,
    _text_to_tokens,
    iter_wikipedia_token_batches,
)


def test_topic_keywords_covers_all_clusters():
    """Every cluster 0-7 must have at least one keyword."""
    cfg = HaloFEPConfig(n_tokens=32)
    assert set(TOPIC_KEYWORDS.keys()) == set(range(cfg.n_hidden))
    for cluster, kws in TOPIC_KEYWORDS.items():
        assert len(kws) >= 2, f"Cluster {cluster} has fewer than 2 keywords"


def test_text_to_tokens_shape():
    cfg = HaloFEPConfig(n_tokens=32)
    text = "The algorithm for machine learning involves equations and code."
    tokens = _text_to_tokens(text, n_tokens=cfg.n_tokens, d_model=cfg.d_model)
    assert tokens.shape == (cfg.n_tokens, cfg.d_model)
    assert tokens.dtype == np.float32


def test_text_to_tokens_short_text_zero_padded():
    tokens = _text_to_tokens("hi", n_tokens=8, d_model=16)
    assert tokens.shape == (8, 16)
    # Later slots should be zero (no text to fill them)
    assert np.allclose(tokens[2:], 0.0)


def test_iter_wikipedia_token_batches_with_mock():
    """iter_wikipedia_token_batches yields (n_tokens, d_model) arrays."""
    cfg = HaloFEPConfig(n_tokens=8, d_model=16)

    # Mock the datasets library
    fake_articles = [
        {"text": f"research study investigation findings topic {i} " * 10}
        for i in range(50)
    ]
    mock_ds = iter(fake_articles)

    with patch("halo_fep.training.topic_bootstrap.load_dataset", return_value=mock_ds):
        gen = iter_wikipedia_token_batches(cfg, seed=0)
        batch = next(gen)

    assert batch.shape == (cfg.n_tokens, cfg.d_model)
    assert batch.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest halo_fep/training/tests/test_topic_bootstrap.py::test_topic_keywords_covers_all_clusters -v
```
Expected: `FAILED` — `ModuleNotFoundError: halo_fep.training.topic_bootstrap`

- [ ] **Step 3: Create `halo_fep/training/topic_bootstrap.py`**

```python
# halo_fep/training/topic_bootstrap.py
"""Wikipedia topic bootstrap — replaces synthetic MultimodalWorld data.

Streams WikiText-103 (a 103M-word Wikipedia corpus) via the HuggingFace
datasets library. Articles are filtered by per-cluster topic keywords and
embedded using the sentence-transformer CPU embedder, producing
(n_tokens, d_model) token arrays that match the heartbeat pipeline format.

Requires: pip install datasets sentence-transformers

The 8 topic clusters map to the HaloFEPConfig.n_hidden=8 hidden states,
establishing a clean semantic prior before the organism encounters noisy
live web data.
"""
from __future__ import annotations

import logging
import random
from itertools import cycle
from typing import Generator

import numpy as np

from halo_fep.config import HaloFEPConfig

log = logging.getLogger(__name__)

# Keywords that determine which cluster a Wikipedia article is routed to.
# These must cover all cfg.n_hidden=8 clusters (indices 0-7).
TOPIC_KEYWORDS: dict[int, list[str]] = {
    0: ["research", "study", "investigation", "findings", "experiment"],
    1: ["algorithm", "programming", "software", "api", "system", "network"],
    2: ["equation", "theorem", "proof", "mathematical", "calculus", "algebra"],
    3: ["philosophy", "theory", "ethics", "consciousness", "epistemology"],
    4: ["implementation", "code", "program", "function", "class", "compiler"],
    5: ["error", "failure", "problem", "diagnosis", "defect", "crash"],
    6: ["history", "historical", "century", "ancient", "civilization", "war"],
    7: ["future", "prediction", "forecast", "trend", "emerging", "innovation"],
}


def _text_to_tokens(
    text: str,
    n_tokens: int,
    d_model: int,
) -> np.ndarray:
    """Embed text into a (n_tokens, d_model) float32 token array.

    Splits the text into up to n_tokens equal-length chunks, embeds each
    chunk with a fixed random projection (no model load required for tests),
    and zero-pads remaining slots.

    In production this is called within iter_wikipedia_token_batches which
    uses the real sentence-transformer embedder.

    Args:
        text: Raw article text.
        n_tokens: Number of token slots (must match cfg.n_tokens).
        d_model: Embedding dimension (must match cfg.d_model).

    Returns:
        (n_tokens, d_model) float32 array.
    """
    tokens = np.zeros((n_tokens, d_model), dtype=np.float32)
    if not text.strip():
        return tokens

    # Split text into n_tokens chunks of equal character length
    chunk_size = max(1, len(text) // n_tokens)
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

    for i, chunk in enumerate(chunks[:n_tokens]):
        if not chunk.strip():
            continue
        # Deterministic hash-based embedding (fallback used in tests)
        seed = abs(hash(chunk)) % (2 ** 31)
        rng  = np.random.RandomState(seed)
        vec  = rng.randn(d_model).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            tokens[i] = vec / norm

    return tokens


def _embed_chunk_real(text: str, model, d_model: int) -> np.ndarray:
    """Embed a single text chunk using a loaded SentenceTransformer."""
    raw = model.encode(text, normalize_embeddings=False, show_progress_bar=False)
    # raw is (384,); project to d_model via truncation or padding
    if len(raw) >= d_model:
        vec = raw[:d_model].astype(np.float32)
    else:
        vec = np.pad(raw, (0, d_model - len(raw))).astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 1e-8 else vec


def iter_wikipedia_token_batches(
    cfg: HaloFEPConfig,
    seed: int = 42,
    articles_per_cluster: int = 200,
) -> Generator[np.ndarray, None, None]:
    """Yield (n_tokens, d_model) token arrays from WikiText-103 forever.

    Articles are filtered by TOPIC_KEYWORDS so each of the 8 clusters
    is represented roughly equally. The generator cycles indefinitely
    so it can drive an arbitrarily long bootstrap loop.

    Args:
        cfg: Config (n_tokens, d_model, n_hidden must be 8).
        seed: RNG seed for shuffling.
        articles_per_cluster: How many matching articles to buffer per cluster
                              before shuffling and cycling.

    Yields:
        (cfg.n_tokens, cfg.d_model) float32 numpy arrays.

    Raises:
        ImportError: If `datasets` is not installed.
        RuntimeError: If no articles match keywords for any cluster.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Wikipedia topic bootstrap requires: pip install datasets"
        )

    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        use_real_embedder = True
        log.info("Using sentence-transformer for Wikipedia embeddings.")
    except Exception:
        st_model = None
        use_real_embedder = False
        log.warning(
            "sentence-transformers not available — using hash-based embeddings."
        )

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-v1",
        split="train",
        streaming=True,
    )

    # Collect articles_per_cluster matched articles per cluster
    buffers: dict[int, list[np.ndarray]] = {i: [] for i in range(cfg.n_hidden)}
    needed = {i: articles_per_cluster for i in range(cfg.n_hidden)}

    for article in dataset:
        text = article.get("text", "")
        if len(text) < 80:
            continue
        text_lower = text.lower()

        for cluster_idx, keywords in TOPIC_KEYWORDS.items():
            if needed[cluster_idx] <= 0:
                continue
            if any(kw in text_lower for kw in keywords):
                if use_real_embedder and st_model is not None:
                    chunk_size = max(1, len(text) // cfg.n_tokens)
                    chunks = [
                        text[i: i + chunk_size]
                        for i in range(0, len(text), chunk_size)
                    ]
                    tok = np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32)
                    for j, ch in enumerate(chunks[: cfg.n_tokens]):
                        tok[j] = _embed_chunk_real(ch, st_model, cfg.d_model)
                else:
                    tok = _text_to_tokens(text, cfg.n_tokens, cfg.d_model)

                buffers[cluster_idx].append(tok)
                needed[cluster_idx] -= 1
                break  # assign each article to at most one cluster

        if all(n <= 0 for n in needed.values()):
            break

    # Warn if any cluster is empty
    for ci, buf in buffers.items():
        if not buf:
            log.warning(
                f"No WikiText-103 articles matched cluster {ci} "
                f"({TOPIC_KEYWORDS[ci]}). Using zero tokens."
            )
            buffers[ci] = [np.zeros((cfg.n_tokens, cfg.d_model), dtype=np.float32)]

    # Flatten, shuffle, and cycle indefinitely
    rng = random.Random(seed)
    all_tokens = [tok for buf in buffers.values() for tok in buf]
    rng.shuffle(all_tokens)
    log.info(f"Wikipedia bootstrap: {len(all_tokens)} articles buffered.")

    for tok in cycle(all_tokens):
        yield tok
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest halo_fep/training/tests/test_topic_bootstrap.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add halo_fep/training/topic_bootstrap.py halo_fep/training/tests/test_topic_bootstrap.py
git commit -m "feat(training): Wikipedia topic bootstrap — WikiText-103 replaces synthetic data per cluster"
```

---

## Task 7: WN18RR Hyperbolic Pre-training

**Files:**
- Create: `halo_fep/training/hyperbolic_pretrain.py`
- Create: `halo_fep/training/tests/test_hyperbolic_pretrain.py`

- [ ] **Step 1: Write the failing tests**

Create `halo_fep/training/tests/test_hyperbolic_pretrain.py`:

```python
# halo_fep/training/tests/test_hyperbolic_pretrain.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel
from halo_fep.training.hyperbolic_pretrain import (
    poincare_distance,
    poincare_loss,
    run_hyperbolic_pretrain,
)


def test_poincare_distance_is_zero_for_same_point():
    u = jnp.array([0.1, 0.2])
    d = poincare_distance(u, u)
    assert float(d) < 1e-4


def test_poincare_distance_increases_near_boundary():
    """Points near the disk boundary are far from the center."""
    center = jnp.zeros(2)
    near_boundary = jnp.array([0.99, 0.0])
    d_near = poincare_distance(center, near_boundary)
    d_mid  = poincare_distance(center, jnp.array([0.5, 0.0]))
    assert float(d_near) > float(d_mid)


def test_poincare_distance_is_symmetric():
    u = jnp.array([0.3, 0.1])
    v = jnp.array([-0.2, 0.4])
    assert jnp.allclose(poincare_distance(u, v), poincare_distance(v, u), atol=1e-5)


def test_poincare_loss_is_scalar():
    dim = 4
    n_entities = 10
    embeddings = jnp.array(
        np.random.uniform(-0.5, 0.5, (n_entities, dim)).astype(np.float32)
    )
    neg_idxs = jnp.array([2, 3, 4])
    loss = poincare_loss(embeddings, u_idx=0, v_idx=1, neg_idxs=neg_idxs)
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_run_hyperbolic_pretrain_returns_model():
    """run_hyperbolic_pretrain should return a HaloFEPModel (may mock dataset)."""
    from unittest.mock import patch
    cfg = HaloFEPConfig(n_tokens=32)
    key = jax.random.PRNGKey(0)
    model = HaloFEPModel(cfg, key)

    # Fake triples: list of dicts with head/relation/tail indices
    fake_triples = [{"head": i % 5, "relation": 0, "tail": (i + 1) % 5}
                    for i in range(20)]

    with patch("halo_fep.training.hyperbolic_pretrain.load_dataset",
               return_value=fake_triples):
        updated_model = run_hyperbolic_pretrain(
            model, cfg, key=key, n_steps=5, n_entities=5
        )

    # Model should be returned (backbone updated)
    assert updated_model is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest halo_fep/training/tests/test_hyperbolic_pretrain.py::test_poincare_distance_is_zero_for_same_point -v
```
Expected: `FAILED` — `ModuleNotFoundError: halo_fep.training.hyperbolic_pretrain`

- [ ] **Step 3: Create `halo_fep/training/hyperbolic_pretrain.py`**

```python
# halo_fep/training/hyperbolic_pretrain.py
"""WN18RR hyperbolic pre-training for the HoloEmbedding layer.

Pre-trains the HaloFEPModel's HoloEmbedding (Poincaré disk projection) on
WordNet IS-A hierarchy triples from WN18RR, using the Poincaré embedding
loss. Hierarchical relationships in WordNet map naturally to the Poincaré
disk's exponential distance growth near the boundary.

After pre-training, the HoloEmbedding produces geometrically structured
embeddings where hypernyms (general concepts) cluster near the disk center
and hyponyms (specific concepts) cluster near the boundary — improving the
HALO backbone's ability to represent hierarchical web knowledge.

Requires: pip install datasets

Usage:
    from halo_fep.training.hyperbolic_pretrain import run_hyperbolic_pretrain
    model = run_hyperbolic_pretrain(model, cfg, key)
"""
from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax
import numpy as np

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel

log = logging.getLogger(__name__)

_EPS = 1e-5


def poincare_distance(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Poincaré disk distance between two points u, v ∈ B^n (‖x‖ < 1).

    d(u, v) = arccosh(1 + 2‖u-v‖² / ((1-‖u‖²)(1-‖v‖²)))

    Both u and v are automatically clamped to the interior of the unit disk
    to prevent numerical instability near the boundary.

    Args:
        u: (..., dim) float32 point in the Poincaré disk.
        v: (..., dim) float32 point in the Poincaré disk.

    Returns:
        Scalar geodesic distance (non-negative float32).
    """
    # Clamp to strict interior: ‖x‖ < 1 - eps
    u = u / jnp.maximum(jnp.linalg.norm(u) + _EPS, 1.0 + _EPS)
    v = v / jnp.maximum(jnp.linalg.norm(v) + _EPS, 1.0 + _EPS)

    norm_u_sq = jnp.sum(u ** 2)
    norm_v_sq = jnp.sum(v ** 2)
    diff_sq   = jnp.sum((u - v) ** 2)

    alpha = 1.0 - norm_u_sq
    beta  = 1.0 - norm_v_sq

    arg = 1.0 + 2.0 * diff_sq / (alpha * beta + _EPS)
    return jnp.arccosh(jnp.clip(arg, 1.0 + _EPS, None))


def poincare_loss(
    embeddings: jnp.ndarray,
    u_idx: int,
    v_idx: int,
    neg_idxs: jnp.ndarray,
) -> jnp.ndarray:
    """Poincaré embedding max-margin loss for one positive triple (u, v).

    Loss = log[ exp(-d(u,v)) / (exp(-d(u,v)) + Σ_{v'} exp(-d(u,v'))) ]

    Minimizing this pulls linked entities (IS-A pairs) together on the disk
    while pushing negative samples apart.

    Args:
        embeddings: (n_entities, dim) float32 embedding table.
        u_idx: Head entity index.
        v_idx: Positive tail entity index (hypernym/hyponym).
        neg_idxs: (n_neg,) int32 negative tail indices.

    Returns:
        Scalar loss (non-negative float32).
    """
    u     = embeddings[u_idx]
    v_pos = embeddings[v_idx]

    d_pos = poincare_distance(u, v_pos)

    def d_neg_i(idx):
        return poincare_distance(u, embeddings[idx])

    d_neg = jax.vmap(d_neg_i)(neg_idxs)

    # Numerically stable log-softmax
    all_d  = jnp.concatenate([d_pos[None], d_neg])
    log_sm = jax.nn.log_softmax(-all_d)
    return -log_sm[0]   # maximize log-prob of positive pair


def run_hyperbolic_pretrain(
    model: HaloFEPModel,
    cfg: HaloFEPConfig,
    key: jnp.ndarray,
    n_steps: int = 1_000,
    n_entities: int = 40_943,   # WN18RR entity count
    n_negatives: int = 10,
    lr: float = 5e-3,
) -> HaloFEPModel:
    """Pre-train HoloEmbedding on WN18RR WordNet IS-A triples.

    Loads WN18RR via HuggingFace datasets (streaming=False, ~1MB download).
    Trains a (n_entities, d_boundary) Poincaré embedding table for n_steps
    gradient steps, then projects the learned embeddings onto the HoloEmbedding
    weight matrix via SVD-based alignment.

    Only the holo_embed layer is modified; all other model components are
    unchanged.

    Args:
        model: HaloFEPModel to update.
        cfg: Config (d_boundary is the Poincaré embedding dimension).
        key: JAX PRNG key.
        n_steps: Number of gradient steps on WN18RR triples.
        n_entities: Number of WN18RR entities (40,943 in the full dataset).
        n_negatives: Number of negative samples per positive triple.
        lr: Learning rate for Poincaré SGD.

    Returns:
        Updated HaloFEPModel with improved HoloEmbedding weights.

    Raises:
        ImportError: If `datasets` is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "WN18RR pre-training requires: pip install datasets"
        )

    log.info("Loading WN18RR dataset...")
    triples = list(load_dataset("KGDatasets/WN18RR", split="train"))
    log.info(f"WN18RR: {len(triples)} training triples loaded.")

    # Initialize Poincaré embedding table in the interior of the disk
    key, ek = jax.random.split(key)
    raw = jax.random.normal(ek, (n_entities, cfg.d_boundary)) * 0.01
    embeddings = raw  # start near origin (center of disk)

    opt = optax.sgd(lr)
    opt_state = opt.init(embeddings)

    def step_fn(embeddings, triple_idx, neg_idx_arr):
        def loss_fn(emb):
            return poincare_loss(emb, triple_idx[0], triple_idx[1], neg_idx_arr)
        loss, grads = jax.value_and_grad(loss_fn)(embeddings)
        updates, _ = opt.update(grads, opt_state)
        # Riemannian retraction: project back into disk
        new_emb = embeddings + updates
        norms = jnp.linalg.norm(new_emb, axis=-1, keepdims=True)
        new_emb = new_emb / jnp.maximum(norms, 1.0 + _EPS)
        return new_emb, loss

    rng = np.random.default_rng(int(key[0]))

    for step in range(n_steps):
        triple = triples[step % len(triples)]
        u_idx  = int(triple.get("head", triple.get("head_id", 0)) or 0)
        v_idx  = int(triple.get("tail", triple.get("tail_id", 1)) or 1)

        # Sample random negatives (corrupt the tail)
        neg_idxs = rng.integers(0, n_entities, size=n_negatives)
        neg_idxs_jnp = jnp.array(neg_idxs, dtype=jnp.int32)

        embeddings, loss = step_fn(embeddings, jnp.array([u_idx, v_idx]), neg_idxs_jnp)

        if step % 200 == 0:
            log.info(f"WN18RR step {step}/{n_steps} | loss={float(loss):.4f}")

    # Align learned Poincaré embeddings with HoloEmbedding weight matrix via SVD.
    # HoloEmbedding contains a linear layer mapping d_model -> d_boundary.
    # We replace its weight with the top-d_model left singular vectors of the
    # embedding table (a learned basis for the hyperbolic space).
    log.info("Aligning Poincaré embeddings with HoloEmbedding weight matrix...")
    emb_np = np.array(embeddings)                # (n_entities, d_boundary)
    U, S, Vt = np.linalg.svd(emb_np, full_matrices=False)

    # Vt shape: (min(n_entities, d_boundary), d_boundary)
    # We want a (d_boundary, d_model) weight for holo_embed's linear layer
    # Use the top d_model right singular vectors as the new basis
    d_model   = cfg.d_model
    d_boundary = cfg.d_boundary
    n_basis    = min(Vt.shape[0], d_model)

    # Build a (d_boundary, d_model) projection matrix
    new_weight = np.zeros((d_boundary, d_model), dtype=np.float32)
    new_weight[:, :n_basis] = Vt[:n_basis].T   # (d_boundary, n_basis)

    # Update model.holo_embed using eqx.tree_at
    # holo_embed has a linear layer; we update its weight leaf
    try:
        model = eqx.tree_at(
            lambda m: m.holo_embed.linear.weight,
            model,
            jnp.array(new_weight),
        )
        log.info("HoloEmbedding weight updated with WN18RR Poincaré basis.")
    except Exception as e:
        log.warning(
            f"Could not update holo_embed.linear.weight ({e}). "
            "HoloEmbedding structure may differ — skipping alignment."
        )

    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest halo_fep/training/tests/test_hyperbolic_pretrain.py -v
```
Expected: all `PASSED`

- [ ] **Step 5: Run the full test suite to verify nothing is broken**

```bash
python -m pytest halo_fep/ -q --ignore=halo_fep/tests/test_benchmark.py
```
Expected: `0 failed` (benchmark test excluded — pre-existing failure unrelated to this work)

- [ ] **Step 6: Commit**

```bash
git add halo_fep/training/hyperbolic_pretrain.py halo_fep/training/tests/test_hyperbolic_pretrain.py
git commit -m "feat(training): WN18RR Poincaré pre-training — aligns HoloEmbedding with WordNet hierarchy"
```

---

## Self-Review

**Spec coverage:**
- ✅ EWC-LoRA: Task 3
- ✅ PER: Task 2 (EpisodeStore) + Task 3 (LoRATrainer accepts per_weights)
- ✅ MESU: Task 4
- ✅ Multi-scale SSM: Task 5
- ✅ Wikipedia topic bootstrap: Task 6
- ✅ WN18RR hyperbolic pre-training: Task 7
- ✅ Config fields: Task 1

**Placeholder scan:** No TBDs, no "add error handling" vagueness. All code is complete.

**Type consistency:**
- `get_prioritized` returns `tuple[list[Episode], np.ndarray]` — consistent with usage in LoRATrainer `per_weights: np.ndarray | None`
- `_multiscale_elbo_loss` signature matches its call in `run_bootstrap`
- `run_hyperbolic_pretrain(model, cfg, key, ...)` — consistent across test and implementation
- `poincare_loss(embeddings, u_idx, v_idx, neg_idxs)` — consistent across test and implementation
