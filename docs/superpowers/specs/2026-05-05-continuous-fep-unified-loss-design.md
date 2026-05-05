# Continuous FEP + Unified ELBO Loss Design

## Goal

Replace the discrete argmax observation bottleneck in ObsBridge with a differentiable softmax output, update belief_update to use a soft log-likelihood, and unify the training objective into a single principled ELBO — eliminating the ad-hoc `lambda_fep` hyperparameter.

---

## Problem Statement

Two coupled deficiencies in the current HALO-FEP integration:

1. **Argmax obs bottleneck** — `ObsBridge.__call__` returns `(N_agents,) int32` via argmax. This severs gradient flow between the HALO backbone and FEP belief dynamics. The ActionBridge feedback loop only works via the carry state; there is no direct gradient path from HALO outputs through FEP beliefs back to the backbone.

2. **Ad-hoc weighted loss** — `L_total = L_HALO + lambda_fep * F_swarm` treats flow matching and free energy as unrelated objectives with a hand-tuned scalar weight. Both are variational bounds on log-evidence for the same underlying generative model; they should be terms in a single ELBO.

---

## Architecture

### Data Flow (changed)

```
tokens (N_tok, d_model)
    │
    ▼
ObsBridge
    │  softmax over logits (was argmax)
    ▼
soft_obs (N_agents, n_obs)  ← float32 probability vector, sums to 1
    │
    ▼
belief_update(mu, soft_obs, gm, cfg)
    │  soft_obs @ log_A replaces A[obs_idx]
    ▼
mu_new (N_agents, n_hidden)

Training signal:
    L_flow  = mean ||v_pred - v_target||²         (flow matching, unchanged)
    L_obs   = -mean einsum('ao,oi,ai', soft_obs, log_A, q_eta)  (new bridge term)
    L_prior = mean KL[q(eta) || D]                (FEP prior)
    L_ELBO  = L_flow + L_obs + L_prior            (no lambda)
```

### Mathematical Basis

Both flow matching and FEP free energy are evidence lower bounds on the same log-evidence `log p(tokens, observations)`:

- `L_flow` bounds `log p(tokens | z)` via the conditional flow matching objective
- `L_obs + L_prior` bounds `log p(observations | tokens)` via the FEP variational free energy

Their sum `L_ELBO` is a valid ELBO on the joint log-evidence. The `lambda_fep` weight disappears because all terms carry equal weight as ELBO components (the trade-off is determined by the model geometry, not a hyperparameter).

`L_obs` is the expected log-likelihood of soft observations under the current belief:

```
L_obs = -E_q[E_o[ln A[o, eta]]]
      = -Σ_agents Σ_eta q(eta) * Σ_o soft_obs[o] * ln A[o, eta]
      = -mean einsum('ao,oi,ai->a', soft_obs, log_A, q_eta)
```

This term forces `q(eta)` and `soft_obs` to agree — exactly the coupling `lambda_fep * F_swarm` was approximating without principled grounding.

---

## Component Changes

### 1. `halo_fep/bridge/obs_bridge.py`

**Change:** Replace `jnp.argmax(...).astype(jnp.int32)` with `jax.nn.softmax(...)`.

Output type: `(N_agents,) int32` → `(N_agents, n_obs) float32`

`_logits()` method stays unchanged (used in NaN test).

### 2. `fep_swarm/agent/belief_update.py`

**Change:** Replace discrete likelihood indexing with soft dot product.

```python
# Before
log_like = jnp.log(gm.A[obs_idx] + 1e-8)  # (n_hidden,)

# After
log_like = soft_obs @ jnp.log(gm.A + 1e-8)  # (n_obs,) @ (n_obs, n_hidden) -> (n_hidden,)
```

Signature change: `obs_idx: jnp.ndarray` (scalar int) → `soft_obs: jnp.ndarray` (shape `(n_obs,)`)

### 3. `halo_fep/model.py`

**Change:** `obs` variable in `halo_fep_step` changes shape from `()` int to `(n_obs,)` float. The `agent_step` inner function receives `s: (n_obs,)` instead of scalar — no logic change needed, only the belief_update call benefits automatically.

Remove the comment `# (N_agents,) int32` and update to `# (N_agents, n_obs) float32`.

### 4. `halo_fep/loss.py` (new file)

New `unified_elbo_loss` function replaces the ad-hoc weighted sum in training scripts.

```python
def unified_elbo_loss(
    model: HaloFEPModel,
    carry: HaloFEPCarry,
    tokens: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, dict]:
    """
    Returns (total_loss, metrics) where metrics has keys:
      l_flow, l_obs, l_prior
    """
    new_carry, (h_out, soft_obs, v_pred, v_target) = halo_fep_step(
        model, carry, tokens, key
    )

    # Flow matching term
    l_flow = jnp.mean((v_pred - v_target) ** 2)

    # Soft observation likelihood (HALO -> FEP bridge)
    q_eta  = jax.nn.softmax(new_carry.swarm_mu)           # (N_agents, n_hidden)
    log_A  = jnp.log(model.gm.A + 1e-8)                  # (n_obs, n_hidden)
    l_obs  = -jnp.mean(
        jnp.einsum('ao,oi,ai->a', soft_obs, log_A, q_eta)
    )

    # KL prior term
    log_q  = jnp.log(q_eta + 1e-8)                       # (N_agents, n_hidden)
    log_D  = jnp.log(jax.nn.softmax(model.gm.log_D) + 1e-8)  # (n_hidden,)
    l_prior = jnp.mean(jnp.sum(q_eta * (log_q - log_D), axis=-1))

    total = l_flow + l_obs + l_prior
    return total, {"l_flow": l_flow, "l_obs": l_obs, "l_prior": l_prior}
```

### 5. `halo_fep/config.py`

**Remove** `lambda_fep` field. Any training script referencing `cfg.lambda_fep` must be updated to use `unified_elbo_loss` directly.

---

## Files Modified

| File | Change type | Description |
|------|-------------|-------------|
| `halo_fep/bridge/obs_bridge.py` | Modify | argmax → softmax, output type change |
| `fep_swarm/agent/belief_update.py` | Modify | soft log-likelihood replaces discrete indexing |
| `halo_fep/model.py` | Modify | comment/annotation update for obs shape |
| `halo_fep/loss.py` | Create | unified_elbo_loss function |
| `halo_fep/config.py` | Modify | remove lambda_fep field |
| `halo_fep/tests/test_obs_bridge.py` | Modify | update shape assertions, add softmax sum test |
| `fep_swarm/tests/test_belief_update.py` | Modify | change obs_idx fixture to soft_obs vector |
| `halo_fep/tests/test_model.py` | Modify | update obs shape assertions |
| `halo_fep/tests/test_loss.py` | Create | unified ELBO loss tests |

---

## Tests

| Test | File | What it verifies |
|------|------|-----------------|
| `test_obs_bridge_soft_output` | `test_obs_bridge.py` | returns `(N_agents, n_obs)` float, rows sum to 1.0 |
| `test_obs_bridge_no_nan` | `test_obs_bridge.py` | unchanged — checks `_logits()` |
| `test_belief_update_soft_obs` | `test_belief_update.py` | soft obs vector produces different mu than uniform prior |
| `test_belief_update_gradient_flows` | `test_belief_update.py` | `jax.grad` w.r.t. `soft_obs` is non-zero |
| `test_unified_elbo_loss_scalar` | `test_loss.py` | loss is a finite scalar |
| `test_unified_elbo_loss_decreases` | `test_loss.py` | loss decreases over 5 gradient steps |
| `test_l_obs_couples_halo_and_fep` | `test_loss.py` | gradient of `L_obs` w.r.t. ObsBridge weights is non-zero |
| `test_halo_fep_step_obs_shape` | `test_model.py` | `obs` output of `halo_fep_step` has shape `(N_agents, n_obs)` |

---

## What Does NOT Change

- `DiscreteGenerativeModel` — matrices A, B, C, D unchanged
- `ActionBridge`, `BeliefBridge` — no changes
- `HALOBackbone`, `HoloEmbedding`, `PageCurveMemory` — no changes
- `HaloFEPCarry` — `swarm_mu` and `swarm_action` shapes unchanged
- `ads_kg_prior`, `v_proj` — no changes
- The EFE per-policy vmap (Phase 1 fix) — unchanged

---

## Migration Note

Training scripts that currently compute:
```python
loss = l_halo + cfg.lambda_fep * f_swarm
```
should be replaced with:
```python
loss, metrics = unified_elbo_loss(model, carry, tokens, key)
```

The `lambda_fep` field can be removed from any config YAML/JSON files.
