# HoloBiont: Phi-3 Removal + 6 GB Scale-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Phi-3.5 LLM dependency and replace with native JAX MetaLayer + HomeostaticRegulator while scaling the HALO backbone and swarm to saturate 6 GB VRAM.

**Architecture:** A hierarchical FEP MetaLayer (eqx.Module, fires every K=10 ticks) reads the swarm's recent belief history and outputs a `log_C` goal update. A HomeostaticRegulator (plain Python class, fires every tick) computes novelty from hidden-state EMA and blends with the MetaLayer output to switch between explore and exploit. Both live entirely in JAX — no LLM, no torch, no HuggingFace.

**Tech Stack:** JAX, Equinox, Optax, FAISS, sentence-transformers, duckduckgo-search

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `halo_fep/config.py` | Modify | Add meta/homeo params; scale defaults; remove `wake_threshold` |
| `halo_fep/intellect/meta_layer.py` | Create | `MetaLayer` eqx.Module + `MetaCarry` dataclass |
| `halo_fep/intellect/homeostatic_regulator.py` | Create | `HomeostaticRegulator` class |
| `halo_fep/model.py` | Modify | Add `meta_layer` field to `HaloFEPModel` |
| `halo_fep/intellect/goal_updater.py` | Modify | Strip to `decay()` only — remove `update_goal` |
| `halo_fep/training/lora_trainer.py` | Modify | Extend `_backbone_filter` to include `meta_layer` |
| `halo_fep/main.py` | Modify | Remove wake cycle; wire MetaLayer + HomeoReg; remove LLM params |
| `halo_fep/intellect/llm_bridge.py` | Delete | Phi-3 gone |
| `halo_fep/intellect/state_compressor.py` | Delete | LLM prompt formatter, no longer needed |
| `halo_fep/tests/test_config.py` | Modify | Remove wake_threshold tests; add meta/homeo param tests |
| `halo_fep/tests/test_meta_layer.py` | Create | Unit tests for MetaLayer |
| `halo_fep/tests/test_homeostatic_regulator.py` | Create | Unit tests for HomeostaticRegulator |
| `halo_fep/tests/test_integration.py` | Modify | Assert no LLM imports; assert MetaLayer fires at tick 10 |
| `requirements.txt` | Modify | Remove torch, transformers, bitsandbytes |

---

## Task 1: Config — Scale Params, Meta/Homeo Params, Remove wake_threshold

**Files:**
- Modify: `halo_fep/config.py`
- Modify: `halo_fep/tests/test_config.py`

- [ ] **Step 1: Write failing tests for new config shape**

Add to the bottom of `halo_fep/tests/test_config.py`:

```python
# -------------------------------------------------------------------------
# Scale-up defaults
# -------------------------------------------------------------------------

def test_default_d_model_is_2048():
    cfg = HaloFEPConfig()
    assert cfg.d_model == 2048

def test_default_n_agents_is_1024():
    cfg = HaloFEPConfig()
    assert cfg.n_agents == 1024

def test_default_n_hidden_is_16():
    cfg = HaloFEPConfig()
    assert cfg.n_hidden == 16

def test_default_n_obs_is_8():
    cfg = HaloFEPConfig()
    assert cfg.n_obs == 8

# -------------------------------------------------------------------------
# Meta-layer params
# -------------------------------------------------------------------------

def test_default_meta_params_exist():
    cfg = HaloFEPConfig()
    assert cfg.meta_k == 10
    assert cfg.meta_n_hidden == 8
    assert cfg.meta_n_actions == 4

def test_meta_k_zero_raises():
    with pytest.raises(ValueError, match="meta_k"):
        HaloFEPConfig(meta_k=0)

def test_meta_n_hidden_zero_raises():
    with pytest.raises(ValueError, match="meta_n_hidden"):
        HaloFEPConfig(meta_n_hidden=0)

# -------------------------------------------------------------------------
# Homeostatic params
# -------------------------------------------------------------------------

def test_default_homeo_params_exist():
    cfg = HaloFEPConfig()
    assert cfg.homeo_ema_alpha == 0.99
    assert cfg.homeo_novelty_threshold_factor == 0.8
    assert cfg.homeo_blend_clip == 1.0

def test_homeo_ema_alpha_zero_raises():
    with pytest.raises(ValueError, match="homeo_ema_alpha"):
        HaloFEPConfig(homeo_ema_alpha=0.0)

def test_homeo_ema_alpha_above_one_raises():
    with pytest.raises(ValueError, match="homeo_ema_alpha"):
        HaloFEPConfig(homeo_ema_alpha=1.1)

# -------------------------------------------------------------------------
# wake_threshold is removed
# -------------------------------------------------------------------------

def test_wake_threshold_does_not_exist():
    cfg = HaloFEPConfig()
    assert not hasattr(cfg, "wake_threshold")
```

- [ ] **Step 2: Delete wake_threshold tests that will now fail**

In `halo_fep/tests/test_config.py`, remove these three functions entirely:
- `test_wake_threshold_zero_raises`
- `test_wake_threshold_negative_raises`

(They will error because the parameter no longer exists.)

- [ ] **Step 3: Run tests to confirm failures**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_config.py -v 2>&1 | tail -30
```

Expected: 10+ failures for missing attributes and missing validation.

- [ ] **Step 4: Update `halo_fep/config.py`**

Replace the entire `HaloFEPConfig` class body with the following (preserve the module docstring and imports):

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HaloFEPConfig:
    """Immutable configuration for the HoloBiont Persistent Mind system."""

    # HALO dims
    d_model: int = 2048
    d_boundary: int = 64
    n_heads: int = 16
    d_head: int = 128
    n_layers: int = 20
    d_state: int = 32
    d_ff: int = 8192
    max_cache: int = 128
    island_size: int = 32
    flow_steps: int = 4
    delta_flow: float = 1.5
    bekenstein_alpha: float = 0.1
    lambda_bek: float = 0.1
    lambda_thermo: float = 0.05
    lambda_page: float = 0.05

    # FEP dims
    n_hidden: int = 16
    n_obs: int = 8
    n_actions: int = 8
    n_policies: int = 16
    tau: int = 3
    inf_steps: int = 16
    inf_lr: float = 0.01
    beta: float = 1.0

    # Swarm
    n_agents: int = 1024
    kappa: float = 0.3
    topology: Literal["all2all", "sparse", "grid"] = "all2all"
    coarse_k: int = 32

    # Bridge
    n_tokens: int = 32

    # Joint training
    lambda_fep: float = 0.1
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42

    # Continual learning
    ewc_lambda: float = 0.1
    per_alpha: float = 0.6
    per_beta: float = 0.4
    use_mesu: bool = False
    mesu_eta: float = 0.01

    # Heartbeat (wake_threshold removed — no LLM wake cycle)
    tick_interval: int = 60

    # Meta-layer (hierarchical FEP)
    meta_n_hidden: int = 8
    meta_n_actions: int = 4
    meta_k: int = 10

    # Homeostatic regulator
    homeo_ema_alpha: float = 0.99
    homeo_novelty_threshold_factor: float = 0.8
    homeo_blend_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.n_agents % self.coarse_k != 0:
            raise ValueError(
                f"n_agents ({self.n_agents}) must be divisible by coarse_k ({self.coarse_k})"
            )
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * d_head ({self.d_head}) must equal "
                f"d_model ({self.d_model}), got {self.n_heads * self.d_head}"
            )
        if self.flow_steps > self.n_layers:
            raise ValueError(
                f"flow_steps ({self.flow_steps}) must be <= n_layers ({self.n_layers})"
            )
        if self.n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {self.n_tokens}")
        if self.tick_interval <= 0:
            raise ValueError(f"tick_interval must be > 0, got {self.tick_interval}")
        if self.n_hidden < 1:
            raise ValueError(f"n_hidden must be >= 1, got {self.n_hidden}")
        if self.n_obs < 1:
            raise ValueError(f"n_obs must be >= 1, got {self.n_obs}")
        if self.n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {self.n_actions}")
        if self.tau < 1:
            raise ValueError(f"tau must be >= 1, got {self.tau}")
        if not (0.0 <= self.ewc_lambda):
            raise ValueError(f"ewc_lambda must be >= 0, got {self.ewc_lambda}")
        if not (0.0 <= self.per_alpha <= 1.0):
            raise ValueError(f"per_alpha must be in [0, 1], got {self.per_alpha}")
        if not (0.0 <= self.per_beta <= 1.0):
            raise ValueError(f"per_beta must be in [0, 1], got {self.per_beta}")
        if self.meta_k < 1:
            raise ValueError(f"meta_k must be >= 1, got {self.meta_k}")
        if self.meta_n_hidden < 1:
            raise ValueError(f"meta_n_hidden must be >= 1, got {self.meta_n_hidden}")
        if self.meta_n_actions < 1:
            raise ValueError(f"meta_n_actions must be >= 1, got {self.meta_n_actions}")
        if not (0.0 < self.homeo_ema_alpha <= 1.0):
            raise ValueError(
                f"homeo_ema_alpha must be in (0, 1], got {self.homeo_ema_alpha}"
            )
```

- [ ] **Step 5: Run config tests to confirm they pass**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_config.py -v 2>&1 | tail -30
```

Expected: All tests pass. The old `test_n_heads_d_head_consistency` test uses `HaloFEPConfig(n_heads=4, d_head=64, d_model=256)` — this still works because the validation only checks consistency, not specific values.

- [ ] **Step 6: Commit**

```bash
cd D:/New_Ai && git add halo_fep/config.py halo_fep/tests/test_config.py && git commit -m "feat(config): scale to 6GB — d_model=2048, 1024 agents, meta/homeo params, remove wake_threshold"
```

---

## Task 2: MetaLayer — TDD

**Files:**
- Create: `halo_fep/tests/test_meta_layer.py`
- Create: `halo_fep/intellect/meta_layer.py`

- [ ] **Step 1: Write failing tests**

Create `halo_fep/tests/test_meta_layer.py`:

```python
"""Tests for MetaLayer — hierarchical FEP meta-controller."""
import pytest
import jax
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.intellect.meta_layer import MetaLayer, MetaCarry

# Small config that satisfies all __post_init__ constraints
_CFG = HaloFEPConfig(
    d_model=64, n_heads=4, d_head=16, n_layers=2, d_ff=128,
    n_agents=32, coarse_k=8, n_tokens=4, tau=1, flow_steps=2,
    n_hidden=4, n_obs=4, n_actions=4, n_policies=4,
    meta_n_hidden=4, meta_n_actions=2, meta_k=3,
)


def _make_layer():
    return MetaLayer(_CFG, jax.random.PRNGKey(0))


def test_init_carry_shape():
    ml = _make_layer()
    carry = ml.init_carry()
    assert isinstance(carry, MetaCarry)
    assert carry.ring_buffer.shape == (3, 4)   # (meta_k=3, n_hidden=4)
    assert carry.meta_mu.shape == (4,)          # (meta_n_hidden=4,)
    assert carry.tick_count == 0


def test_step_no_fire_before_k():
    """log_C must be None for ticks < meta_k."""
    ml = _make_layer()
    carry = ml.init_carry()
    mean_belief = jnp.ones(4) / 4.0
    for i in range(2):   # meta_k=3, so tick 1 and 2 must not fire
        carry, log_C = ml.step(carry, mean_belief, fe=1.0)
        assert log_C is None, f"Should not fire at tick {i+1}"
    assert carry.tick_count == 2


def test_step_fires_at_k():
    """log_C must not be None exactly at tick meta_k."""
    ml = _make_layer()
    carry = ml.init_carry()
    mean_belief = jnp.ones(4) / 4.0
    for i in range(3):   # fires at tick 3
        carry, log_C = ml.step(carry, mean_belief, fe=1.0)
    assert log_C is not None
    assert log_C.shape == (4,)   # (n_obs=4,)


def test_log_C_is_valid_log_prob():
    """All log_C entries must be <= 0 and finite."""
    ml = _make_layer()
    carry = ml.init_carry()
    mean_belief = jnp.ones(4) / 4.0
    for _ in range(3):
        carry, log_C = ml.step(carry, mean_belief, fe=1.0)
    assert jnp.all(log_C <= 1e-5), "log_C entries must be <= 0"
    assert jnp.all(jnp.isfinite(log_C)), "log_C must have no NaN/Inf"


def test_meta_mu_changes_after_meta_step():
    """meta_mu must differ from initial zeros after a meta-step fires."""
    ml = _make_layer()
    carry = ml.init_carry()
    initial_mu = carry.meta_mu
    mean_belief = jnp.ones(4) / 4.0
    for _ in range(3):
        carry, _ = ml.step(carry, mean_belief, fe=1.0)
    assert not jnp.allclose(carry.meta_mu, initial_mu)


def test_ring_buffer_fills_correctly():
    """Each step must write mean_belief into the ring buffer at the right index."""
    ml = _make_layer()
    carry = ml.init_carry()
    beliefs = [
        jnp.array([0.1, 0.2, 0.3, 0.4]),
        jnp.array([0.4, 0.3, 0.2, 0.1]),
        jnp.array([0.25, 0.25, 0.25, 0.25]),
    ]
    for i, b in enumerate(beliefs):
        carry, _ = ml.step(carry, b, fe=1.0)
    for i, b in enumerate(beliefs):
        assert jnp.allclose(carry.ring_buffer[i], b, atol=1e-6), \
            f"ring_buffer[{i}] mismatch"


def test_fires_again_at_2k():
    """MetaLayer must fire at tick 6 as well (2 * meta_k)."""
    ml = _make_layer()
    carry = ml.init_carry()
    mean_belief = jnp.ones(4) / 4.0
    fire_count = 0
    for _ in range(6):
        carry, log_C = ml.step(carry, mean_belief, fe=1.0)
        if log_C is not None:
            fire_count += 1
    assert fire_count == 2, f"Expected 2 fires (at tick 3 and 6), got {fire_count}"
```

- [ ] **Step 2: Run to confirm ImportError (file doesn't exist yet)**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_meta_layer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'halo_fep.intellect.meta_layer'`

- [ ] **Step 3: Implement `halo_fep/intellect/meta_layer.py`**

Create `halo_fep/intellect/meta_layer.py`:

```python
"""MetaLayer — hierarchical FEP controller operating at slow timescale.

Fires every meta_k ticks. Reads accumulated belief history from a ring
buffer, runs variational inference on a small meta generative model, and
outputs a log_C update for the main model's generative model.

The goal_vectors matrix (meta_n_hidden, n_obs) is the only learnable
component trained during nightly dreaming. Each row is a candidate log_C
bias vector. The meta-belief softmax weights their combination.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.agent.belief_update import belief_update


@dataclass(frozen=True)
class _MetaGMCfg:
    """Minimal config accepted by DiscreteGenerativeModel and belief_update."""
    n_hidden: int
    n_obs: int
    n_actions: int
    n_policies: int = 4
    tau: int = 1
    inf_steps: int = 8
    inf_lr: float = 0.01
    beta: float = 1.0


@dataclass
class MetaCarry:
    """Non-JAX carry for the MetaLayer (lives in HeartbeatLoop, not in JIT)."""
    ring_buffer: jnp.ndarray   # (meta_k, n_hidden) — recent mean beliefs
    meta_mu: jnp.ndarray       # (meta_n_hidden,) — current meta-belief logits
    tick_count: int            # Python int, incremented each tick


class MetaLayer(eqx.Module):
    """Hierarchical FEP meta-layer (eqx.Module — trained during nightly dreaming).

    Parameters are included in the LoRATrainer trainable mask alongside
    the backbone so they improve with experience.
    """
    gm: DiscreteGenerativeModel   # small meta generative model
    goal_vectors: jnp.ndarray     # (meta_n_hidden, n_obs) learnable goal bias
    _meta_k: int = eqx.field(static=True)
    _meta_n_hidden: int = eqx.field(static=True)
    _meta_n_actions: int = eqx.field(static=True)
    _n_hidden: int = eqx.field(static=True)
    _n_obs: int = eqx.field(static=True)
    _inf_steps: int = eqx.field(static=True)
    _inf_lr: float = eqx.field(static=True)

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        k1, k2 = jax.random.split(key)
        meta_gm_cfg = _MetaGMCfg(
            n_hidden=cfg.meta_n_hidden,
            n_obs=cfg.n_hidden,          # meta-observations = main belief state dim
            n_actions=cfg.meta_n_actions,
        )
        self.gm = DiscreteGenerativeModel(meta_gm_cfg, k1)
        self.goal_vectors = jax.random.normal(k2, (cfg.meta_n_hidden, cfg.n_obs))
        self._meta_k = cfg.meta_k
        self._meta_n_hidden = cfg.meta_n_hidden
        self._meta_n_actions = cfg.meta_n_actions
        self._n_hidden = cfg.n_hidden
        self._n_obs = cfg.n_obs
        self._inf_steps = cfg.inf_steps
        self._inf_lr = cfg.inf_lr

    def init_carry(self) -> MetaCarry:
        """Return a zeroed MetaCarry for a fresh run."""
        return MetaCarry(
            ring_buffer=jnp.zeros((self._meta_k, self._n_hidden)),
            meta_mu=jnp.zeros(self._meta_n_hidden),
            tick_count=0,
        )

    def step(
        self,
        carry: MetaCarry,
        mean_belief: jnp.ndarray,   # (n_hidden,) softmax-normalised swarm mean
        fe: float,
    ) -> tuple[MetaCarry, Optional[jnp.ndarray]]:
        """Process one tick.

        Returns
        -------
        (new_carry, log_C_meta) where log_C_meta is None unless a meta-step fires.
        log_C_meta shape: (n_obs,) — valid log-probability vector (all entries <= 0).
        """
        idx = carry.tick_count % self._meta_k
        new_ring = carry.ring_buffer.at[idx].set(mean_belief)
        new_tick = carry.tick_count + 1

        if new_tick % self._meta_k != 0:
            return MetaCarry(
                ring_buffer=new_ring,
                meta_mu=carry.meta_mu,
                tick_count=new_tick,
            ), None

        # --- Meta-step fires ---
        # Reduce ring buffer: mean across K ticks → (n_hidden,) meta-observation
        meta_obs = jnp.mean(new_ring, axis=0)             # (n_hidden,)
        meta_obs_prob = jax.nn.softmax(meta_obs)          # valid probability vector

        # Variational inference on meta generative model
        meta_cfg = _MetaGMCfg(
            n_hidden=self._meta_n_hidden,
            n_obs=self._n_hidden,
            n_actions=self._meta_n_actions,
            inf_steps=self._inf_steps,
            inf_lr=self._inf_lr,
        )
        new_meta_mu = belief_update(carry.meta_mu, meta_obs_prob, self.gm, meta_cfg)

        # Compute log_C: weighted combination of learnable goal vectors
        meta_probs = jax.nn.softmax(new_meta_mu)          # (meta_n_hidden,)
        log_C_raw = meta_probs @ self.goal_vectors         # (n_obs,)
        log_C_meta = jax.nn.log_softmax(log_C_raw)        # valid log-probs, all <= 0

        return MetaCarry(
            ring_buffer=new_ring,
            meta_mu=new_meta_mu,
            tick_count=new_tick,
        ), log_C_meta
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_meta_layer.py -v 2>&1 | tail -20
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai && git add halo_fep/intellect/meta_layer.py halo_fep/tests/test_meta_layer.py && git commit -m "feat(meta_layer): hierarchical FEP meta-controller — fires every K ticks, outputs log_C goal update"
```

---

## Task 3: HomeostaticRegulator — TDD

**Files:**
- Create: `halo_fep/tests/test_homeostatic_regulator.py`
- Create: `halo_fep/intellect/homeostatic_regulator.py`

- [ ] **Step 1: Write failing tests**

Create `halo_fep/tests/test_homeostatic_regulator.py`:

```python
"""Tests for HomeostaticRegulator — novelty-driven explore/exploit."""
import jax
import jax.numpy as jnp
import numpy as np

from halo_fep.config import HaloFEPConfig
from halo_fep.intellect.homeostatic_regulator import HomeostaticRegulator

_CFG = HaloFEPConfig(
    d_model=64, n_heads=4, d_head=16, n_layers=2, d_ff=128,
    n_agents=32, coarse_k=8, n_tokens=4, tau=1, flow_steps=2,
    n_hidden=4, n_obs=4, n_actions=4, n_policies=4,
    meta_n_hidden=4, meta_n_actions=2, meta_k=3,
)


def _zeros_h():
    """Repeated identical hidden state — zero novelty after warmup."""
    return jnp.zeros((_CFG.n_tokens, _CFG.d_model))


def _random_h(seed: int):
    return jax.random.normal(jax.random.PRNGKey(seed), (_CFG.n_tokens, _CFG.d_model))


def test_output_shape():
    reg = HomeostaticRegulator(_CFG)
    novelty, log_C = reg.update(_zeros_h(), recent_episodes=[])
    assert isinstance(novelty, float)
    assert log_C.shape == (_CFG.n_obs,)


def test_log_C_is_finite():
    reg = HomeostaticRegulator(_CFG)
    _, log_C = reg.update(_zeros_h(), recent_episodes=[])
    assert jnp.all(jnp.isfinite(log_C))


def test_explore_mode_returns_uniform():
    """Highly novel inputs should produce uniform (log-uniform) log_C."""
    reg = HomeostaticRegulator(_CFG)
    # Feed very diverse inputs to drive novelty high above EMA threshold
    for seed in range(20):
        novelty, log_C = reg.update(_random_h(seed), recent_episodes=[])

    expected = jnp.full((_CFG.n_obs,), -jnp.log(_CFG.n_obs))
    # At least some of the 20 steps should have been in explore mode
    # Final state: novelty_ema has settled — feed one more very novel input
    novel_h = jax.random.normal(jax.random.PRNGKey(999), (_CFG.n_tokens, _CFG.d_model)) * 100.0
    novelty, log_C = reg.update(novel_h, recent_episodes=[])
    assert jnp.allclose(log_C, expected, atol=1e-5), \
        f"Expected uniform log_C in explore mode, got {log_C}"


def test_ema_buffers_update_after_first_call():
    """h_mean must change from its initial zeros after the first update."""
    reg = HomeostaticRegulator(_CFG)
    h_mean_before = reg.h_mean.copy()
    reg.update(_random_h(0), recent_episodes=[])
    assert not jnp.allclose(reg.h_mean, h_mean_before), \
        "h_mean must update after first call"


def test_novelty_ema_updates():
    reg = HomeostaticRegulator(_CFG)
    before = reg.novelty_ema
    reg.update(_random_h(0), recent_episodes=[])
    assert reg.novelty_ema != before


def test_no_nan_after_repeated_identical_inputs():
    """Stable identical inputs must not produce NaN in any output."""
    reg = HomeostaticRegulator(_CFG)
    h = _zeros_h()
    for _ in range(50):
        novelty, log_C = reg.update(h, recent_episodes=[])
    assert not jnp.any(jnp.isnan(log_C))
    assert not jnp.isnan(novelty)
```

- [ ] **Step 2: Run to confirm ImportError**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_homeostatic_regulator.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'halo_fep.intellect.homeostatic_regulator'`

- [ ] **Step 3: Implement `halo_fep/intellect/homeostatic_regulator.py`**

Create `halo_fep/intellect/homeostatic_regulator.py`:

```python
"""HomeostaticRegulator — novelty-driven explore/exploit log_C controller.

Runs every tick. Maintains running EMA of HALO hidden states.

* novelty > adaptive_threshold  →  EXPLORE: uniform log_C (maximum entropy)
* novelty ≤ adaptive_threshold  →  EXPLOIT: log_C biased toward most
                                    successful recent observation cluster

No trainable parameters — pure EMA math. State lives in HeartbeatLoop.
"""
from __future__ import annotations

import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig


class HomeostaticRegulator:
    """Novelty-driven explore/exploit controller.

    Parameters
    ----------
    cfg : HaloFEPConfig — uses d_model, n_obs, homeo_ema_alpha,
          homeo_novelty_threshold_factor, homeo_blend_clip.
    """

    def __init__(self, cfg: HaloFEPConfig) -> None:
        self.cfg = cfg
        self.h_mean = jnp.zeros(cfg.d_model)
        self.h_var = jnp.ones(cfg.d_model)
        self.novelty_ema: float = 1.0   # start high — unexperienced organism

    def update(
        self,
        h_out: jnp.ndarray,        # (n_tokens, d_model) backbone output
        recent_episodes: list,
    ) -> tuple[float, jnp.ndarray]:
        """Update EMA state and return (novelty_score, log_C_homeo).

        Parameters
        ----------
        h_out           : (n_tokens, d_model) HALO hidden states from current tick.
        recent_episodes : List of recent Episode objects (may be empty).

        Returns
        -------
        novelty   : Scalar float — normalised Mahalanobis novelty score.
        log_C_homeo : (n_obs,) log-probability vector for goal preference.
        """
        h_now = jnp.mean(h_out, axis=0)    # (d_model,) mean over tokens

        # Normalised novelty: how far h_now is from running mean
        novelty = float(jnp.mean(
            (h_now - self.h_mean) ** 2 / (self.h_var + 1e-8)
        ))

        # Update running EMA statistics
        alpha = self.cfg.homeo_ema_alpha
        new_h_mean = alpha * self.h_mean + (1.0 - alpha) * h_now
        diff = h_now - new_h_mean
        self.h_mean = new_h_mean
        self.h_var = alpha * self.h_var + (1.0 - alpha) * diff ** 2
        self.novelty_ema = alpha * self.novelty_ema + (1.0 - alpha) * novelty

        threshold = self.cfg.homeo_novelty_threshold_factor * self.novelty_ema

        if novelty > threshold:
            log_C_homeo = self._explore_log_C()
        else:
            log_C_homeo = self._exploit_log_C(recent_episodes)

        return novelty, log_C_homeo

    def _explore_log_C(self) -> jnp.ndarray:
        """Uniform log_C — equal preference over all observations."""
        n = self.cfg.n_obs
        return jnp.full((n,), -jnp.log(n))

    def _exploit_log_C(self, recent_episodes: list) -> jnp.ndarray:
        """Concentrate log_C on the observation cluster with lowest free_energy_delta."""
        n = self.cfg.n_obs
        if not recent_episodes:
            return self._explore_log_C()

        best = min(recent_episodes, key=lambda ep: ep.free_energy_delta)
        mu = jnp.array(best.swarm_mu)               # (n_agents, n_hidden)
        mean_mu = jnp.mean(mu, axis=0)              # (n_hidden,)
        belief_idx = int(jnp.argmax(mean_mu))
        obs_idx = belief_idx % n

        # Concentrated: near-zero probability everywhere except preferred obs
        log_C = jnp.full((n,), -10.0)
        return log_C.at[obs_idx].set(0.0)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_homeostatic_regulator.py -v 2>&1 | tail -20
```

Expected: All 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai && git add halo_fep/intellect/homeostatic_regulator.py halo_fep/tests/test_homeostatic_regulator.py && git commit -m "feat(homeostatic_regulator): novelty-driven explore/exploit log_C controller"
```

---

## Task 4: Model — Add MetaLayer to HaloFEPModel

**Files:**
- Modify: `halo_fep/model.py`

- [ ] **Step 1: Write a failing test**

Add to `halo_fep/tests/test_model.py` (append to existing file):

```python
def test_model_has_meta_layer():
    """HaloFEPModel must have a meta_layer attribute of type MetaLayer."""
    from halo_fep.intellect.meta_layer import MetaLayer
    cfg = HaloFEPConfig(
        d_model=64, n_heads=4, d_head=16, n_layers=2, d_ff=128,
        n_agents=32, coarse_k=8, n_tokens=4, tau=1, flow_steps=2,
        n_hidden=4, n_obs=4, n_actions=4, n_policies=4,
        meta_n_hidden=4, meta_n_actions=2, meta_k=3,
    )
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))
    assert hasattr(model, "meta_layer")
    assert isinstance(model.meta_layer, MetaLayer)
```

Run it:
```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_model.py::test_model_has_meta_layer -v 2>&1 | tail -10
```

Expected: FAIL — `HaloFEPModel has no attribute 'meta_layer'`

- [ ] **Step 2: Add `meta_layer` to `HaloFEPModel` in `halo_fep/model.py`**

In `halo_fep/model.py`, add the import at the top (after existing imports):

```python
from halo_fep.intellect.meta_layer import MetaLayer
```

In the `HaloFEPModel` class body, add the field declaration after `v_proj`:

```python
    v_proj:        eqx.nn.Linear  # d_boundary -> d_model, for KG prior projection
    meta_layer:    MetaLayer      # hierarchical FEP meta-controller
    cfg:           HaloFEPConfig = eqx.field(static=True)
```

In `HaloFEPModel.__init__`, keys[7] is currently unused. Add:

```python
        self.v_proj        = eqx.nn.Linear(cfg.d_boundary, cfg.d_model, use_bias=False, key=keys[5])
        self.gm            = DiscreteGenerativeModel(cfg, keys[6])
        self.meta_layer    = MetaLayer(cfg, keys[7])
```

- [ ] **Step 3: Run the new test and full model test suite**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_model.py -v 2>&1 | tail -20
```

Expected: All tests pass including the new `test_model_has_meta_layer`.

- [ ] **Step 4: Commit**

```bash
cd D:/New_Ai && git add halo_fep/model.py halo_fep/tests/test_model.py && git commit -m "feat(model): add MetaLayer to HaloFEPModel"
```

---

## Task 5: GoalUpdater — Strip to decay() Only

**Files:**
- Modify: `halo_fep/intellect/goal_updater.py`

- [ ] **Step 1: Replace `goal_updater.py` with decay-only version**

Overwrite `halo_fep/intellect/goal_updater.py` with:

```python
# halo_fep/intellect/goal_updater.py
"""GoalUpdater — decays model.gm.log_C toward uniform each tick.

The update_goal(text) method has been removed. Goal-setting is now handled
by MetaLayer (hierarchical FEP) and HomeostaticRegulator. This class only
performs the per-tick decay to prevent goal fixation.
"""
from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel


class GoalUpdater:
    def __init__(self, cfg: HaloFEPConfig) -> None:
        self.cfg = cfg

    def decay(self, model: HaloFEPModel, alpha: float = 0.99) -> HaloFEPModel:
        """Decay log_C 1% toward uniform each tick.

        Prevents the system from fixating on a single goal cluster forever.
        """
        n_obs = self.cfg.n_obs
        uniform = jnp.full((n_obs,), -jnp.log(n_obs))
        new_log_c = alpha * model.gm.log_C + (1.0 - alpha) * uniform
        return eqx.tree_at(lambda m: m.gm.log_C, model, new_log_c)
```

- [ ] **Step 2: Run existing tests to check nothing broke**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/ -v -k "not test_integration" 2>&1 | tail -20
```

Expected: All tests pass. (Any test that called `update_goal` will have been in the LLM integration path which we're removing.)

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai && git add halo_fep/intellect/goal_updater.py && git commit -m "refactor(goal_updater): strip to decay-only — LLM update_goal removed"
```

---

## Task 6: LoRATrainer — Extend Trainable Filter to Include MetaLayer

**Files:**
- Modify: `halo_fep/training/lora_trainer.py`

- [ ] **Step 1: Update `_backbone_filter` to include `meta_layer`**

In `halo_fep/training/lora_trainer.py`, find the `_backbone_filter` function (lines ~59-67) and replace it:

```python
def _backbone_filter(model: HaloFEPModel):
    """Return a boolean pytree: True for backbone AND meta_layer leaves.

    Backbone: core HALO SSM + attention parameters.
    MetaLayer: goal_vectors and meta generative model — trained alongside backbone
               so goal-setting improves with experience.
    All other parameters (bridges, embeddings, etc.) are frozen.
    """
    false_model   = jtu.tree_map(lambda _: False, model)
    true_backbone = jtu.tree_map(lambda _: True, model.backbone)
    true_meta     = jtu.tree_map(lambda _: True, model.meta_layer)
    result = eqx.tree_at(lambda m: m.backbone, false_model, true_backbone)
    return eqx.tree_at(lambda m: m.meta_layer, result, true_meta)
```

- [ ] **Step 2: Run trainer tests**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_lora_trainer.py -v 2>&1 | tail -20
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai && git add halo_fep/training/lora_trainer.py && git commit -m "feat(lora_trainer): include meta_layer in trainable filter — goal vectors trained during nightly dreaming"
```

---

## Task 7: Main — Remove Wake Cycle, Wire MetaLayer + HomeostaticRegulator

**Files:**
- Modify: `halo_fep/main.py`

- [ ] **Step 1: Replace `halo_fep/main.py` with the updated version**

Overwrite `halo_fep/main.py` with:

```python
# halo_fep/main.py
"""Persistent Mind heartbeat orchestrator.

Runs the subconscious tick loop indefinitely. Free energy is now managed
entirely by the native JAX MetaLayer (slow timescale) and
HomeostaticRegulator (fast timescale) — no external LLM.

Architecture per tick
---------------------
1.  Perception          web fetch + embed → (n_tokens, d_model)
2.  HALO+FEP step       update carry → h_out, soft_obs, v_pred, v_target
3.  Free-energy         compute scalar F
4.  HomeostaticReg      novelty from h_out EMA → log_C_homeo
5.  MetaLayer           every meta_k ticks → log_C_meta (else None)
6.  Blend               log_C_final → model.gm.log_C
7.  FEP matrix update   EMA on A, B, D
8.  Goal decay          log_C decays 1% toward uniform
9.  Episode store       persist to SQLite + FAISS
10. Nightly dreaming    02:00-02:15 local time
"""
from __future__ import annotations

import collections
import datetime
import logging
import signal
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPModel, halo_fep_step
from halo_fep.memory.schema import Episode
from halo_fep.utils import compute_free_energy
from halo_fep.paths import ensure_dirs, EPISODE_DIR, BOOTSTRAP_CKPT

log = logging.getLogger(__name__)

_RECENT_EPS_MAXLEN = 50   # deque size for homeostatic exploit branch


def _is_nightly_window() -> bool:
    now = datetime.datetime.now()
    return now.hour == 2 and now.minute < 15


class HeartbeatLoop:
    """Encapsulates one run of the subconscious heartbeat loop.

    Parameters
    ----------
    cfg              : System configuration.
    model            : Initialised HaloFEPModel (includes meta_layer).
    perception       : PerceptionPipeline — web fetch + embedding.
    memory           : EpisodeStore — SQLite + FAISS memory.
    goal_updater     : GoalUpdater — decay only (optional).
    fep_updater      : FEPUpdater — EMA updates to A, B, D (optional).
    lora_trainer     : LoRATrainer — nightly fine-tuning (optional).
    """

    def __init__(
        self,
        cfg: HaloFEPConfig,
        model: HaloFEPModel,
        perception,
        memory,
        goal_updater=None,
        fep_updater=None,
        lora_trainer=None,
    ) -> None:
        from halo_fep.intellect.homeostatic_regulator import HomeostaticRegulator
        self.cfg              = cfg
        self.model            = model
        self.carry            = model.init_carry(jax.random.PRNGKey(cfg.seed))
        self.meta_carry       = model.meta_layer.init_carry()
        self.perception       = perception
        self.memory           = memory
        self.goal_updater     = goal_updater
        self.fep_updater      = fep_updater
        self.lora_trainer     = lora_trainer
        self.homeostatic_reg  = HomeostaticRegulator(cfg)
        self._prev_fe: float | None = None
        self._nightly_done_date: str | None = None
        self._recent_eps: collections.deque = collections.deque(maxlen=_RECENT_EPS_MAXLEN)

    def tick(self) -> None:
        """Run one subconscious tick. Never raises — logs errors and returns."""
        # --- 1. Perception ---
        query = self.perception.query_from_beliefs(self.carry)
        try:
            tokens = self.perception.embed(query)
        except Exception as e:
            log.warning(f"Perception failed: {e}. Skipping tick.")
            return

        # --- 2. HALO+FEP step ---
        key, carry_key = jax.random.split(self.carry.key)
        self.carry = self.carry._replace(key=key)
        try:
            self.carry, (h_out, soft_obs, v_pred, v_target) = halo_fep_step(
                self.model, self.carry, tokens, carry_key
            )
        except Exception as e:
            log.error(f"halo_fep_step failed: {e}. Skipping tick.")
            return

        # --- 3. Free-energy ---
        fe = float(compute_free_energy(self.carry, self.model))
        if not jnp.isfinite(fe):
            log.error("NaN/Inf in free energy — skipping tick.")
            return
        fe_delta      = (fe - self._prev_fe) if self._prev_fe is not None else 0.0
        self._prev_fe = fe

        # --- 4. HomeostaticRegulator ---
        try:
            novelty, log_C_homeo = self.homeostatic_reg.update(
                h_out, list(self._recent_eps)
            )
        except Exception as e:
            log.warning(f"HomeostaticRegulator failed: {e}")
            novelty = 0.0
            n_obs = self.cfg.n_obs
            log_C_homeo = jnp.full((n_obs,), -jnp.log(n_obs))

        # --- 5. MetaLayer (fires every meta_k ticks) ---
        mean_belief = jnp.mean(
            jax.nn.softmax(self.carry.swarm_mu, axis=-1), axis=0
        )   # (n_hidden,)
        try:
            self.meta_carry, log_C_meta = self.model.meta_layer.step(
                self.meta_carry, mean_belief, fe
            )
        except Exception as e:
            log.warning(f"MetaLayer.step failed: {e}")
            log_C_meta = None

        # --- 6. Blend log_C and write to model ---
        try:
            novelty_w = float(jnp.clip(
                jnp.array(novelty / (self.homeostatic_reg.novelty_ema + 1e-8)),
                0.0, self.cfg.homeo_blend_clip,
            ))
            w = novelty_w / (novelty_w + 1.0)   # sigmoid-like in [0, 1]
            if log_C_meta is not None:
                log_C_final = w * log_C_homeo + (1.0 - w) * log_C_meta
            else:
                log_C_final = log_C_homeo
            self.model = eqx.tree_at(lambda m: m.gm.log_C, self.model, log_C_final)
        except Exception as e:
            log.warning(f"log_C blend failed: {e}")

        # --- 7. FEP matrix update ---
        if self.fep_updater is not None:
            try:
                self.model = self.fep_updater.update(
                    self.model, self.carry, None, soft_obs
                )
            except Exception as e:
                log.warning(f"FEP matrix update failed: {e}")

        # --- 8. Goal decay ---
        if self.goal_updater is not None:
            self.model = self.goal_updater.decay(self.model)

        # --- 9. Episode persistence ---
        query_embed_np = self.perception.embed_query(query)
        episode = Episode(
            query             = query,
            tokens            = jnp.array(tokens).__array__(),
            swarm_mu          = jnp.array(self.carry.swarm_mu).__array__(),
            free_energy       = fe,
            free_energy_delta = fe_delta,
        )
        self.memory.add(episode, query_embed=query_embed_np)
        self._recent_eps.append(episode)
        log.info(
            f"Tick | query={query!r} | FE={fe:.3f} | "
            f"FE_delta={fe_delta:+.3f} | novelty={novelty:.3f}"
        )

        # --- 10. Nightly dreaming ---
        today = datetime.date.today().isoformat()
        if _is_nightly_window() and self._nightly_done_date != today:
            success = self._nightly_learning()
            if success:
                self._nightly_done_date = today

    def _nightly_learning(self) -> bool:
        log.info("Nightly learning cycle starting.")
        if self.lora_trainer is None:
            return True
        try:
            episodes = self.memory.get_high_confidence()
            if not episodes:
                log.info("No high-confidence episodes for nightly training.")
                return True
            self.model, info = self.lora_trainer.run(self.model, episodes)
            log.info(f"Nightly learning done: {info}")
            self.memory.flush()
            return True
        except Exception as e:
            log.error(f"Nightly learning failed: {e}")
            return False


def main() -> None:
    """Entry point: configure logging, build all components, run heartbeat."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_dirs()

    cfg = HaloFEPConfig(tick_interval=60)

    try:
        from halo_fep.training.bootstrap import load_checkpoint
        model = load_checkpoint(cfg, BOOTSTRAP_CKPT)
    except Exception:
        log.info("No checkpoint found — initialising fresh model.")
        model = HaloFEPModel(cfg, jax.random.PRNGKey(cfg.seed))

    from halo_fep.perception.pipeline  import PerceptionPipeline
    from halo_fep.memory.episode_store import EpisodeStore
    from halo_fep.intellect.goal_updater import GoalUpdater
    from halo_fep.training.fep_updater  import FEPUpdater
    from halo_fep.training.lora_trainer import LoRATrainer

    memory = EpisodeStore(str(EPISODE_DIR))

    loop = HeartbeatLoop(
        cfg          = cfg,
        model        = model,
        perception   = PerceptionPipeline(cfg),
        memory       = memory,
        goal_updater = GoalUpdater(cfg),
        fep_updater  = FEPUpdater(cfg),
        lora_trainer = LoRATrainer(cfg),
    )

    shutdown_requested = False

    def _handle_signal(sig, frame):
        nonlocal shutdown_requested
        log.info(f"Received signal {sig} — initiating graceful shutdown.")
        shutdown_requested = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log.info("Heartbeat started. Press Ctrl+C to stop.")
    while not shutdown_requested:
        tick_start = time.time()
        loop.tick()
        elapsed    = time.time() - tick_start
        sleep_time = max(0.0, cfg.tick_interval - elapsed)
        if elapsed > 1.5 * cfg.tick_interval:
            log.warning(
                f"Tick overrun: {elapsed:.2f}s "
                f"(threshold {1.5 * cfg.tick_interval:.2f}s)"
            )
        time.sleep(sleep_time)

    memory.flush()
    log.info("Heartbeat loop exited cleanly.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the unit test suite (excluding integration tests)**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/ -v -k "not test_integration" 2>&1 | tail -30
```

Expected: All unit tests pass.

- [ ] **Step 3: Commit**

```bash
cd D:/New_Ai && git add halo_fep/main.py && git commit -m "feat(main): remove LLM wake cycle — wire MetaLayer + HomeostaticRegulator, native JAX goal-setting"
```

---

## Task 8: Cleanup — Delete LLM Files + Update Requirements

**Files:**
- Delete: `halo_fep/intellect/llm_bridge.py`
- Delete: `halo_fep/intellect/state_compressor.py`
- Modify: `requirements.txt` (or equivalent)

- [ ] **Step 1: Delete the LLM files**

```bash
cd D:/New_Ai && git rm halo_fep/intellect/llm_bridge.py halo_fep/intellect/state_compressor.py
```

- [ ] **Step 2: Confirm no remaining imports of deleted modules**

```bash
cd D:/New_Ai && grep -r "llm_bridge\|state_compressor\|LLMBridge\|StateCompressor" halo_fep/ --include="*.py"
```

Expected: No output (zero matches).

- [ ] **Step 3: Check requirements file exists and update it**

```bash
cd D:/New_Ai && ls requirements*.txt 2>/dev/null || ls pyproject.toml 2>/dev/null
```

If `requirements.txt` exists, remove the lines for `torch`, `transformers`, `bitsandbytes`:

```bash
cd D:/New_Ai && grep -n "torch\|transformers\|bitsandbytes" requirements*.txt
```

Remove those lines from the file using Edit tool, or if none exist, confirm they are absent:

```bash
cd D:/New_Ai && python -c "import halo_fep.main" 2>&1
```

Expected: No ImportError.

- [ ] **Step 4: Confirm torch/transformers not imported at runtime**

```bash
cd D:/New_Ai && python -c "
import sys
import halo_fep.main
import halo_fep.model
import halo_fep.intellect.goal_updater
import halo_fep.intellect.meta_layer
import halo_fep.intellect.homeostatic_regulator
llm_imports = [m for m in sys.modules if 'torch' in m or 'transformers' in m or 'bitsandbytes' in m]
print('LLM modules in sys.modules:', llm_imports)
assert not llm_imports, f'Unexpected LLM imports: {llm_imports}'
print('PASS: no LLM imports at module load time')
"
```

Expected: `PASS: no LLM imports at module load time`

- [ ] **Step 5: Commit**

```bash
cd D:/New_Ai && git add -u && git commit -m "chore: delete llm_bridge + state_compressor, remove torch/transformers/bitsandbytes deps"
```

---

## Task 9: Integration Test — Assert New Architecture Invariants

**Files:**
- Modify: `halo_fep/tests/test_integration.py`

- [ ] **Step 1: Read the current integration test**

Open `halo_fep/tests/test_integration.py` and locate any existing tick-loop tests.

- [ ] **Step 2: Add new integration assertions**

Append to `halo_fep/tests/test_integration.py`:

```python
# ---------------------------------------------------------------------------
# Architecture invariants after Phi-3 removal
# ---------------------------------------------------------------------------

def _make_small_loop():
    """Build a HeartbeatLoop with mocked perception and memory for testing."""
    import collections
    from unittest.mock import MagicMock
    import numpy as np
    import jax
    import jax.numpy as jnp

    from halo_fep.config import HaloFEPConfig
    from halo_fep.model import HaloFEPModel
    from halo_fep.main import HeartbeatLoop
    from halo_fep.memory.schema import Episode

    cfg = HaloFEPConfig(
        d_model=64, n_heads=4, d_head=16, n_layers=2, d_ff=128,
        n_agents=32, coarse_k=8, n_tokens=4, tau=1, flow_steps=2,
        n_hidden=4, n_obs=4, n_actions=4, n_policies=4,
        meta_n_hidden=4, meta_n_actions=2, meta_k=10,
        tick_interval=1,
    )
    model = HaloFEPModel(cfg, jax.random.PRNGKey(0))

    # Mock perception: returns fixed zero token tensor
    perception = MagicMock()
    perception.query_from_beliefs.return_value = "test query"
    perception.embed.return_value = jnp.zeros((cfg.n_tokens, cfg.d_model))
    perception.embed_query.return_value = np.zeros(256, dtype=np.float32)

    # Mock memory
    memory = MagicMock()
    memory.get_high_confidence.return_value = []

    loop = HeartbeatLoop(cfg=cfg, model=model, perception=perception, memory=memory)
    return loop, cfg


def test_no_llm_imports_in_runtime():
    """torch, transformers, bitsandbytes must never be imported by the tick path."""
    import sys
    import halo_fep.main
    import halo_fep.model
    import halo_fep.intellect.meta_layer
    import halo_fep.intellect.homeostatic_regulator
    import halo_fep.intellect.goal_updater

    bad = [m for m in sys.modules if m.split(".")[0] in {"torch", "transformers", "bitsandbytes"}]
    assert not bad, f"LLM runtime imports found: {bad}"


def test_meta_layer_fires_at_tick_10():
    """MetaLayer must fire exactly once at tick 10, not before."""
    loop, cfg = _make_small_loop()

    meta_fires = []
    original_step = loop.model.meta_layer.step

    def tracked_step(carry, mean_belief, fe):
        new_carry, log_C = original_step(carry, mean_belief, fe)
        if log_C is not None:
            meta_fires.append(carry.tick_count + 1)
        return new_carry, log_C

    loop.model.meta_layer.step = tracked_step  # monkey-patch for tracking

    for _ in range(15):
        loop.tick()

    assert 10 in meta_fires, f"MetaLayer must fire at tick 10, fires were: {meta_fires}"
    assert meta_fires.count(10) == 1, "MetaLayer must fire exactly once at tick 10"


def test_log_C_changes_after_meta_step():
    """model.gm.log_C must differ from its initial value after tick 10."""
    import jax.numpy as jnp
    loop, cfg = _make_small_loop()
    initial_log_C = loop.model.gm.log_C.copy()

    for _ in range(11):   # run past the meta_k=10 boundary
        loop.tick()

    assert not jnp.allclose(loop.model.gm.log_C, initial_log_C), \
        "log_C must change after MetaLayer fires"


def test_loop_has_no_llm_attribute():
    """HeartbeatLoop must not have an llm attribute."""
    loop, _ = _make_small_loop()
    assert not hasattr(loop, "llm"), "HeartbeatLoop must not hold an LLM reference"
    assert not hasattr(loop, "state_compressor"), \
        "HeartbeatLoop must not hold a StateCompressor reference"
```

- [ ] **Step 3: Run integration tests**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/test_integration.py -v 2>&1 | tail -30
```

Expected: All integration tests pass.

- [ ] **Step 4: Run full test suite**

```bash
cd D:/New_Ai && python -m pytest halo_fep/tests/ -v 2>&1 | tail -40
```

Expected: All tests pass (no failures).

- [ ] **Step 5: Final commit**

```bash
cd D:/New_Ai && git add halo_fep/tests/test_integration.py && git commit -m "test(integration): assert MetaLayer fires at tick 10, no LLM imports, log_C changes"
```

---

## Self-Review Checklist

| Spec requirement | Task |
|---|---|
| Remove torch/transformers/bitsandbytes | Task 8 |
| Add MetaLayer (hierarchical FEP, K=10) | Tasks 2, 4 |
| Add HomeostaticRegulator (novelty EMA) | Task 3 |
| log_C blending (novelty_weight) | Task 7 |
| Scale d_model 1024→2048 | Task 1 |
| Scale n_layers 12→20 | Task 1 |
| Scale n_agents 256→1024 | Task 1 |
| Scale n_hidden 8→16, n_obs 4→8 | Task 1 |
| Remove wake_threshold | Task 1 |
| Remove _wake_cycle from main | Task 7 |
| MetaLayer in LoRATrainer trainable mask | Task 6 |
| GoalUpdater stripped to decay only | Task 5 |
| Delete llm_bridge.py, state_compressor.py | Task 8 |
| Unit tests for MetaLayer | Task 2 |
| Unit tests for HomeostaticRegulator | Task 3 |
| Integration test for new arch invariants | Task 9 |
