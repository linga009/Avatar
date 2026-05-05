# Continuous FEP + Unified ELBO Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the discrete argmax observation bottleneck with a differentiable softmax output and unify `L_HALO + lambda_fep * F_swarm` into a single principled ELBO objective.

**Architecture:** `ObsBridge.__call__` returns `(N_agents, n_obs) float32` via softmax instead of `(N_agents,) int32` via argmax. `belief_update` uses `soft_obs @ log_A` instead of discrete indexing `A[obs_idx]`. A new `unified_elbo_loss` function in `loss.py` computes `L_flow + L_obs + L_prior` in one pass, eliminating `lambda_fep`.

**Tech Stack:** JAX, Equinox, Optax, pytest, chex

---

## File Map

| File | Change |
|------|--------|
| `halo_fep/bridge/obs_bridge.py` | `__call__`: argmax → softmax; update docstring |
| `halo_fep/tests/test_bridge.py` | Fix shape test; replace int-range test with sum-to-one |
| `fep_swarm/agent/belief_update.py` | `free_energy` + `belief_update`: `obs_idx: int` → `soft_obs: jnp.ndarray`; soft log-likelihood |
| `fep_swarm/tests/test_agent.py` | Update `obs_idx` fixtures; add gradient flow test |
| `halo_fep/model.py` | Update `obs` comment (1 line) |
| `halo_fep/loss.py` | Add `unified_elbo_loss` function |
| `halo_fep/tests/test_model.py` | Fix obs shape test; replace `_joint_loss` with `unified_elbo_loss`; add coupling gradient test |
| `halo_fep/config.py` | Remove `lambda_fep` field |

---

## Task 1: ObsBridge — argmax to softmax

**Files:**
- Modify: `halo_fep/bridge/obs_bridge.py:30-36`
- Modify: `halo_fep/tests/test_bridge.py:11-21`

- [ ] **Step 1: Write failing tests**

Replace lines 11–21 of `halo_fep/tests/test_bridge.py`:

```python
def test_obs_bridge_output_shape():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    soft_obs = bridge(h_out)
    assert soft_obs.shape == (cfg.n_agents, cfg.n_obs)

def test_obs_bridge_rows_sum_to_one():
    bridge = ObsBridge(cfg, key)
    h_out = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    soft_obs = bridge(h_out)
    row_sums = jnp.sum(soft_obs, axis=-1)  # (N_agents,)
    assert jnp.allclose(row_sums, jnp.ones(cfg.n_agents), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest halo_fep/tests/test_bridge.py::test_obs_bridge_output_shape halo_fep/tests/test_bridge.py::test_obs_bridge_rows_sum_to_one -v
```

Expected: FAIL — `test_obs_bridge_output_shape` fails because shape is `(n_agents,)` not `(n_agents, n_obs)`.

- [ ] **Step 3: Implement — change argmax to softmax in obs_bridge.py**

Replace lines 1–36 of `halo_fep/bridge/obs_bridge.py` with:

```python
# halo_fep/bridge/obs_bridge.py
"""ObsBridge — maps HALO backbone output to per-agent soft observations.

Each agent has a learned soft assignment over N_tok tokens. The assignment
is row-wise softmax normalized (initialized uniform 1/N_tok). A linear
head maps the pooled d_model embedding to n_obs logits; softmax produces
a differentiable probability vector over observations.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from halo_fep.config import HaloFEPConfig


class ObsBridge(eqx.Module):
    assignment_logits: jnp.ndarray  # (N_agents, N_tok) — raw (softmax applied in fwd)
    w_obs: eqx.nn.Linear            # d_model -> n_obs

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        # Initialize assignment uniform (logits = 0 -> softmax = uniform)
        self.assignment_logits = jnp.zeros((cfg.n_agents, cfg.n_tokens))
        self.w_obs = eqx.nn.Linear(cfg.d_model, cfg.n_obs, key=key)

    def _logits(self, h_out: jnp.ndarray) -> jnp.ndarray:
        """Return (N_agents, n_obs) logits — differentiable, before softmax."""
        assignment = jax.nn.softmax(self.assignment_logits, axis=-1)  # (N_agents, N_tok)
        h_pooled   = assignment @ h_out                               # (N_agents, d_model)
        return jax.vmap(self.w_obs)(h_pooled)                        # (N_agents, n_obs)

    def __call__(self, h_out: jnp.ndarray) -> jnp.ndarray:
        """Args:
            h_out: (N_tok, d_model) — HALO backbone output
        Returns:
            soft_obs: (N_agents, n_obs) float32 — differentiable observation probabilities
        """
        return jax.nn.softmax(self._logits(h_out), axis=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest halo_fep/tests/test_bridge.py -v
```

Expected: all bridge tests pass (5 passing).

- [ ] **Step 5: Commit**

```bash
git add halo_fep/bridge/obs_bridge.py halo_fep/tests/test_bridge.py
git commit -m "feat: ObsBridge returns soft (N_agents, n_obs) float32 instead of argmax int32"
```

---

## Task 2: belief_update — discrete index to soft log-likelihood

**Files:**
- Modify: `fep_swarm/agent/belief_update.py`
- Modify: `fep_swarm/tests/test_agent.py:45-65`

- [ ] **Step 1: Write failing tests**

Replace lines 45–65 of `fep_swarm/tests/test_agent.py`:

```python
def test_free_energy_scalar(cfg, gm):
    mu = jnp.zeros(cfg.n_hidden)
    soft_obs = jnp.zeros(cfg.n_obs).at[0].set(1.0)  # one-hot at index 0
    F = free_energy(mu, soft_obs=soft_obs, gm=gm)
    assert F.shape == ()
    assert not jnp.isnan(F)


def test_free_energy_decreases_over_steps(cfg, gm):
    mu = jax.random.normal(jax.random.PRNGKey(5), (cfg.n_hidden,))
    soft_obs = jnp.zeros(cfg.n_obs).at[0].set(1.0)
    F_init = free_energy(mu, soft_obs, gm)
    mu_updated = belief_update(mu, soft_obs, gm, cfg)
    F_final = free_energy(mu_updated, soft_obs, gm)
    assert F_final < F_init


def test_belief_update_no_nan(cfg, gm):
    mu = jnp.zeros(cfg.n_hidden)
    soft_obs = jnp.zeros(cfg.n_obs).at[1].set(1.0)  # one-hot at index 1
    mu_out = belief_update(mu, soft_obs=soft_obs, gm=gm, cfg=cfg)
    assert not jnp.any(jnp.isnan(mu_out))
    chex.assert_shape(mu_out, (cfg.n_hidden,))


def test_belief_update_gradient_flows_through_soft_obs(cfg, gm):
    mu = jnp.zeros(cfg.n_hidden)
    soft_obs = jax.nn.softmax(jnp.ones(cfg.n_obs))  # uniform soft obs
    grad_fn = jax.grad(free_energy, argnums=1)
    g = grad_fn(mu, soft_obs, gm)
    assert g.shape == (cfg.n_obs,)
    assert jnp.any(g != 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest fep_swarm/tests/test_agent.py::test_free_energy_scalar fep_swarm/tests/test_agent.py::test_belief_update_gradient_flows_through_soft_obs -v
```

Expected: FAIL — `free_energy` still takes `obs_idx: int`.

- [ ] **Step 3: Implement — update belief_update.py**

Replace the entire file `fep_swarm/agent/belief_update.py` with:

```python
import jax
import jax.numpy as jnp
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.config import FEPConfig


def free_energy(
    mu: jnp.ndarray,
    soft_obs: jnp.ndarray,
    gm: DiscreteGenerativeModel,
) -> jnp.ndarray:
    """
    F(μ, o) = KL[Q(η;μ) || P(η)] − E_Q[ln P(o|η)]
    mu:       [n_hidden] log-unnormalized beliefs
    soft_obs: [n_obs]    differentiable observation probability vector
    """
    q_eta = jax.nn.softmax(mu)                              # Q(η;μ): [n_hidden]
    p_eta = gm.D                                             # P(η):   [n_hidden]
    kl = jnp.sum(q_eta * (jnp.log(q_eta + 1e-8) - jnp.log(p_eta + 1e-8)))

    # E_Q[ln P(o|η)] = Σ_η Q(η) · (Σ_o soft_obs[o] · ln A[o, η])
    log_A_s = soft_obs @ jnp.log(gm.A + 1e-8)              # (n_obs,) @ (n_obs, n_hidden) -> [n_hidden]
    expected_log_lik = jnp.sum(q_eta * log_A_s)

    return kl - expected_log_lik


def belief_update(
    mu_init: jnp.ndarray,
    soft_obs: jnp.ndarray,
    gm: DiscreteGenerativeModel,
    cfg: FEPConfig,
) -> jnp.ndarray:
    """
    Gradient descent on F: μ_{t+1} = μ_t − lr · ∇_μ F(μ_t, o)
    Uses jax.lax.fori_loop for JIT compatibility.
    """
    grad_F = jax.grad(free_energy)

    def step(i, mu):
        return mu - cfg.inf_lr * grad_F(mu, soft_obs, gm)

    return jax.lax.fori_loop(0, cfg.inf_steps, step, mu_init)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest fep_swarm/tests/test_agent.py -v
```

Expected: all agent tests pass (9 passing).

- [ ] **Step 5: Commit**

```bash
git add fep_swarm/agent/belief_update.py fep_swarm/tests/test_agent.py
git commit -m "feat: belief_update takes soft_obs (n_obs,) float32 instead of obs_idx int"
```

---

## Task 3: model.py — update obs comment

**Files:**
- Modify: `halo_fep/model.py:107`

- [ ] **Step 1: Update comment on the obs line**

In `halo_fep/model.py`, line 107 currently reads:

```python
    obs = model.obs_bridge(h_out)   # (N_agents,) int32
```

Change it to:

```python
    obs = model.obs_bridge(h_out)   # (N_agents, n_obs) float32
```

No other logic changes — `jax.vmap(agent_step)((carry.swarm_mu, carry.swarm_action), obs)` already correctly slices `obs (N_agents, n_obs)` to `s: (n_obs,)` per agent.

- [ ] **Step 2: Verify JIT still compiles**

```
pytest halo_fep/tests/test_model.py::test_closed_loop_jit_compiles -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add halo_fep/model.py
git commit -m "chore: update obs comment to (N_agents, n_obs) float32"
```

---

## Task 4: loss.py — add unified_elbo_loss

**Files:**
- Modify: `halo_fep/loss.py` (append new function)

- [ ] **Step 1: Append unified_elbo_loss to halo_fep/loss.py**

Add the following to the end of `halo_fep/loss.py` (after the `halo_loss` function):

```python

def unified_elbo_loss(
    model,
    carry,
    tokens: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple:
    """Unified ELBO: L_ELBO = L_flow + L_obs + L_prior.

    L_flow  = mean ||v_pred - v_target||^2           (flow matching)
    L_obs   = -mean einsum('ao,oi,ai->a',             (HALO->FEP bridge)
                           soft_obs, log_A, q_eta)
    L_prior = mean KL[q(eta) || D]                   (FEP prior)

    Replaces the ad-hoc L_HALO + lambda_fep * F_swarm.
    Returns (total_loss, {"l_flow": ..., "l_obs": ..., "l_prior": ...}).
    """
    from halo_fep.model import halo_fep_step  # local to avoid circular import at module level

    new_carry, (h_out, soft_obs, v_pred, v_target) = halo_fep_step(
        model, carry, tokens, key
    )

    # Flow matching term
    l_flow = jnp.mean((v_pred - v_target) ** 2)

    # Soft observation likelihood — bridges HALO output to FEP beliefs
    # soft_obs: (N_agents, n_obs), log_A: (n_obs, n_hidden), q_eta: (N_agents, n_hidden)
    q_eta = jax.nn.softmax(new_carry.swarm_mu)            # (N_agents, n_hidden)
    log_A = jnp.log(model.gm.A + 1e-8)                   # (n_obs, n_hidden)
    l_obs = -jnp.mean(
        jnp.einsum('ao,oi,ai->a', soft_obs, log_A, q_eta)
    )

    # KL prior term
    log_q = jnp.log(q_eta + 1e-8)                         # (N_agents, n_hidden)
    log_D = jnp.log(jax.nn.softmax(model.gm.log_D) + 1e-8)  # (n_hidden,)
    l_prior = jnp.mean(jnp.sum(q_eta * (log_q - log_D), axis=-1))

    total = l_flow + l_obs + l_prior
    return total, {"l_flow": l_flow, "l_obs": l_obs, "l_prior": l_prior}
```

- [ ] **Step 2: Verify halo_loss tests still pass (no regression)**

```
pytest halo_fep/tests/test_model.py::test_halo_loss_is_scalar halo_fep/tests/test_model.py::test_halo_loss_no_nan halo_fep/tests/test_model.py::test_halo_loss_parts_keys -v
```

Expected: 3 passing.

- [ ] **Step 3: Commit**

```bash
git add halo_fep/loss.py
git commit -m "feat: add unified_elbo_loss — L_flow + L_obs + L_prior replaces lambda-weighted sum"
```

---

## Task 5: test_model.py — update for new APIs

**Files:**
- Modify: `halo_fep/tests/test_model.py`

This task replaces the `_joint_loss`-based tests (which used the old `free_energy(mu, obs_idx, ...)` discrete API) with `unified_elbo_loss`-based tests, and fixes the `obs` shape assertion.

- [ ] **Step 1: Write new test_model.py**

Replace the entire file `halo_fep/tests/test_model.py` with:

```python
# halo_fep/tests/test_model.py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from halo_fep.config import HaloFEPConfig
from halo_fep.loss import halo_loss, unified_elbo_loss
from halo_fep.model import HaloFEPModel, HaloFEPCarry, halo_fep_step

cfg = HaloFEPConfig()
key = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# halo_loss (HALO-only, unchanged)
# ---------------------------------------------------------------------------

def test_halo_loss_is_scalar():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    total, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert total.shape == ()


def test_halo_loss_no_nan():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    total, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert not jnp.isnan(total)
    for v in parts.values():
        assert not jnp.isnan(v)


def test_halo_loss_parts_keys():
    v_pred       = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    v_target     = jax.random.normal(jax.random.PRNGKey(1), (cfg.n_tokens, cfg.d_model))
    attn_w       = jnp.ones((cfg.n_tokens, cfg.n_tokens)) / cfg.n_tokens
    evict_scores = jax.nn.softmax(jnp.ones(cfg.n_tokens))
    _, parts = halo_loss(v_pred, v_target, attn_w, evict_scores, cfg)
    assert set(parts.keys()) == {"fm", "bek", "thermo", "page"}


# ---------------------------------------------------------------------------
# HaloFEPModel + halo_fep_step
# ---------------------------------------------------------------------------

def _make_model_and_carry():
    model = HaloFEPModel(cfg, key)
    carry = model.init_carry(key)
    return model, carry


def test_model_init():
    model, carry = _make_model_and_carry()
    assert carry.swarm_mu.shape     == (cfg.n_agents, cfg.n_hidden)
    assert carry.swarm_action.shape == (cfg.n_agents, cfg.n_actions)


def test_closed_loop_step_shape():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    assert h_out.shape    == (cfg.n_tokens, cfg.d_model)
    assert obs.shape      == (cfg.n_agents, cfg.n_obs)   # soft float32, not int32
    assert v_pred.shape   == (cfg.n_tokens, cfg.d_model)
    assert v_target.shape == (cfg.n_tokens, cfg.d_model)


def test_closed_loop_step_no_nan():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    new_carry, (h_out, obs, v_pred, v_target) = halo_fep_step(model, carry, tokens, key)
    assert not jnp.any(jnp.isnan(h_out))
    assert not jnp.any(jnp.isnan(obs))
    assert not jnp.any(jnp.isnan(v_pred))


def test_closed_loop_jit_compiles():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    jit_step = eqx.filter_jit(halo_fep_step)
    new_carry, outputs = jit_step(model, carry, tokens, key)
    assert outputs[0].shape == (cfg.n_tokens, cfg.d_model)


# ---------------------------------------------------------------------------
# unified_elbo_loss
# ---------------------------------------------------------------------------

def test_unified_elbo_loss_is_scalar():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    loss, parts = unified_elbo_loss(model, carry, tokens, key)
    assert loss.shape == ()
    assert set(parts.keys()) == {"l_flow", "l_obs", "l_prior"}


def test_unified_elbo_loss_no_nan():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    loss, parts = unified_elbo_loss(model, carry, tokens, key)
    assert not jnp.isnan(loss)
    for v in parts.values():
        assert not jnp.isnan(v)


def test_unified_elbo_loss_gradients_nonzero():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    grad_fn = eqx.filter_grad(unified_elbo_loss, has_aux=True)
    grads, _ = grad_fn(model, carry, tokens, key)
    leaf_grads = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    nonzero = [jnp.any(g != 0.0) for g in leaf_grads if g is not None]
    assert any(nonzero)


def test_l_obs_couples_halo_and_fep():
    """Gradient of L_obs w.r.t. ObsBridge weights must be non-zero.
    This proves end-to-end differentiability through the HALO->FEP interface.
    """
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))

    def l_obs_only(model):
        _, parts = unified_elbo_loss(model, carry, tokens, key)
        return parts["l_obs"]

    grads = eqx.filter_grad(l_obs_only)(model)
    obs_bridge_grads = jax.tree_util.tree_leaves(
        eqx.filter(grads.obs_bridge, eqx.is_array)
    )
    assert any(jnp.any(g != 0.0) for g in obs_bridge_grads)


def test_train_step_does_not_diverge():
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    opt    = optax.adam(cfg.lr)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = opt.init(params)

    def step(params, opt_state, carry):
        model_ = eqx.combine(params, static)
        # Differentiate only the scalar loss; advance carry separately
        loss_fn = lambda m: unified_elbo_loss(m, carry, tokens, key)[0]
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model_)
        new_carry, _ = halo_fep_step(model_, carry, tokens, key)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        return eqx.apply_updates(params, updates), new_opt_state, new_carry, loss

    loss0, _ = unified_elbo_loss(model, carry, tokens, key)
    params, opt_state, carry, _ = step(params, opt_state, carry)
    for _ in range(9):
        params, opt_state, carry, _ = step(params, opt_state, carry)
    loss10, _ = unified_elbo_loss(eqx.combine(params, static), carry, tokens, key)
    assert loss10 < loss0 * 1.5  # not diverging


def test_action_probs_not_uniform_after_belief_update():
    """After belief update with observations, action probs should not be uniform.
    A uniform distribution (all 0.25 for n_actions=4) indicates the EFE bug.
    This test verifies the per-policy EFE fix is working.
    """
    model, carry = _make_model_and_carry()
    tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    k = key
    for _ in range(3):
        k, k_step = jax.random.split(k)
        carry, _ = halo_fep_step(model, carry, tokens, k_step)
    uniform = jnp.ones(cfg.n_actions) / cfg.n_actions
    max_deviation = jnp.max(jnp.abs(carry.swarm_action - uniform))
    assert float(max_deviation) > 1e-6, (
        f"Action probs look uniform (max deviation {float(max_deviation):.2e}). "
        "EFE per-policy fix may not be working."
    )
```

- [ ] **Step 2: Run full test_model.py suite**

```
pytest halo_fep/tests/test_model.py -v
```

Expected: 14 passing, 0 failing.

- [ ] **Step 3: Commit**

```bash
git add halo_fep/tests/test_model.py
git commit -m "test: update test_model to use unified_elbo_loss and soft obs shape"
```

---

## Task 6: config.py — remove lambda_fep

**Files:**
- Modify: `halo_fep/config.py:45`

- [ ] **Step 1: Remove lambda_fep from HaloFEPConfig**

In `halo_fep/config.py`, delete line 45:

```python
    lambda_fep: float = 0.1
```

The field `lr`, `n_steps`, and `seed` remain. The `lambda_bek`, `lambda_thermo`, `lambda_page` fields remain (they are used by `halo_loss`, not `unified_elbo_loss`).

- [ ] **Step 2: Run full test suite**

```
pytest halo_fep/ fep_swarm/ -v
```

Expected: all tests pass, 0 references to `lambda_fep` remain.

- [ ] **Step 3: Verify no stale lambda_fep references**

```
grep -r "lambda_fep" halo_fep/ fep_swarm/
```

Expected: no output (zero matches).

- [ ] **Step 4: Commit**

```bash
git add halo_fep/config.py
git commit -m "feat: remove lambda_fep — unified_elbo_loss needs no coupling hyperparameter"
```

---

## Final verification

- [ ] **Run entire test suite**

```
pytest halo_fep/ fep_swarm/ -v
```

Expected: all tests pass.
