# HALO + FEP-Swarm Integration Design

## 1. Goal

Build a unified closed-loop architecture (`halo_fep/`) that couples HALO (holographic AdS-learned multimodal perception) with FEP-Swarm (multi-agent Active Inference) into a single JAX/equinox system. HALO provides holographically-structured perception; FEP-Swarm provides collective belief inference and action selection; a bidirectional bridge closes the loop so agent actions modulate HALO's attention and agent beliefs condition HALO's generative flow.

Success criterion: a trained model achieves ≥80% goal-inference accuracy on the Multimodal Goal Inference benchmark over 100 episodes.

---

## 2. Background

### 2.1 HALO (existing, PyTorch)

- **Files:** `halo/` — PyTorch + PyTorch Lightning
- **Core:** HoloEmbedding (Poincaré half-space lifting), HoloAttention (bulk-to-boundary K_Δ kernel), SimpleSSM (diagonal scan), HALOBackbone ([S,S,S,H,S,S,S,H]), AdS-KG flow prior, PageCurveMemory (island eviction)
- **Training:** flow matching loss `L_FM + λ_bek·L_Bek + λ_thermo·L_thermo + λ_page·L_page`
- **Dims:** `d_model=256`, `d_boundary=64`, `n_heads=4`, `n_layers=8`

### 2.2 FEP-Swarm (existing, JAX)

- **Files:** `fep_swarm/` — JAX + equinox + diffrax, 49 tests passing
- **Core:** DiscreteGenerativeModel (A/B/C/D), belief update (`μ̇ = −∇_μF` via `jax.lax.fori_loop`), EFE action selection (`G(π) = pragmatic + epistemic`), swarm vmap (N=256 agents), macro coarse-graining + Jacobian eigenanalysis
- **Dims:** `n_hidden=8`, `n_obs=4`, `n_actions=4`, `n_agents=256`

### 2.3 Integration Strategy

**Approach: Bridge coupling.** HALO and FEP-Swarm remain as two distinct modules, connected by a thin bidirectional bridge. Both are ported/kept in JAX for a single JIT-compilable closed loop. The existing `fep_swarm/` is imported unchanged; `halo/` (PyTorch) is ported to `halo_fep/halo_jax/` (equinox). A joint Adam optimizer minimizes `L_HALO + λ·F_swarm`.

---

## 3. Architecture

### 3.1 Module Structure

```
halo_fep/
├── config.py                  # HaloFEPConfig — merged hyperparameters
├── halo_jax/                  # HALO ported to equinox (JAX)
│   ├── holo_embedding.py      # HoloEmbedding (eqx.Module)
│   ├── holo_attention.py      # HoloAttention K_Δ kernel + action bias input
│   ├── simple_ssm.py          # SimpleSSM diagonal scan via jax.lax.scan
│   ├── backbone.py            # HALOBackbone [S,S,S,H,S,S,S,H]
│   ├── ads_kg_prior.py        # AdS-KG flow prior (pure JAX function) + belief conditioning
│   └── page_memory.py         # PageCurveMemory ring buffer (JIT-safe)
├── bridge/
│   ├── obs_bridge.py          # h_out (N_tok, d_model) → (N_agents, n_obs)
│   ├── action_bridge.py       # (N_agents, n_actions) → K_Δ boundary bias (N_tok, d_boundary)
│   └── belief_bridge.py       # (N_agents, n_hidden) → flow conditioning (N_tok, d_model)
├── model.py                   # HaloFEPModel — closed-loop step, eqx.filter_jit
├── loss.py                    # L_total = L_HALO + lambda_fep * F_swarm
├── benchmark/
│   ├── multimodal_world.py    # Synthetic image+text goal-inference task
│   └── eval.py                # Episode runner, success metric
└── tests/
    ├── test_halo_jax.py       # Port correctness vs PyTorch reference
    ├── test_bridge.py         # Shape + gradient-flow tests
    ├── test_model.py          # Closed-loop step, no NaN, joint loss shape
    └── test_benchmark.py      # Behavioral and physics tests
```

**Existing modules unchanged:** `fep_swarm/` is imported as a library. `halo/` (PyTorch) is kept for reference and numerical validation.

### 3.2 Closed-Loop Data Flow

```
[image + text tokens]  (N_tok, d_model)
        │
   HALO backbone (JAX/equinox)
    ← delta_x from action_bridge   (K_Δ boundary bias)
    ← delta_v from belief_bridge   (flow conditioning)
        │ h_out: (N_tok, d_model)
        ▼
   obs_bridge ──────────────────── s_i: (N_agents, n_obs)
                                        │
                           FEP belief update  μ̇ = −∇F
                                        │ μ_i: (N_agents, n_hidden)
                           FEP action selection G(π)
                                        │ a_i: (N_agents, n_actions)
                                       / \
                action_bridge           belief_bridge
                     │                      │
          Δx: boundary bias        Δv: flow conditioning
                     └──────────────────────┘
                                │
                          next HALO step
```

### 3.3 HaloFEPConfig

```python
@dataclass(frozen=True)
class HaloFEPConfig:
    # HALO dims
    d_model: int = 256
    d_boundary: int = 64
    n_heads: int = 4
    n_layers: int = 8
    d_state: int = 16
    d_ff: int = 512
    max_cache: int = 128
    island_size: int = 32
    flow_steps: int = 4
    delta_flow: float = 1.5
    bekenstein_alpha: float = 0.1
    lambda_bek: float = 0.1
    lambda_thermo: float = 0.05
    lambda_page: float = 0.05

    # FEP dims
    n_hidden: int = 8
    n_obs: int = 4
    n_actions: int = 4
    n_policies: int = 8
    tau: int = 3
    inf_steps: int = 16
    inf_lr: float = 0.01
    beta: float = 1.0

    # Swarm
    n_agents: int = 256
    kappa: float = 0.3
    topology: Literal["all2all", "sparse", "grid"] = "all2all"
    coarse_k: int = 16
    eig_gap: float = 10.0

    # Bridge
    n_tokens: int = 48   # N_text + N_image tokens fed to HALO

    # Joint training
    lambda_fep: float = 0.1
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42
```

---

## 4. HALO-JAX Port (`halo_fep/halo_jax/`)

Each PyTorch component is ported to `eqx.Module`. All components must be JIT-compilable.

### 4.1 HoloEmbedding

Two `eqx.nn.Linear` layers (`x_proj: d_model → d_boundary`, `z_proj: d_model → 1`). Depth `z = sigmoid(z_proj(h)) ∈ (0,1)`. No state.

### 4.2 HoloAttention

K_Δ kernel with per-head learnable `log_delta` (`eqx.field`). Action bias `delta_x: (N_tok, d_boundary)` is an additive input argument (not a parameter). Heads are vmapped via `jax.vmap` — no Python loop.

```python
# biased boundary positions
x_biased = x + delta_x                                    # (N_tok, d_boundary)
# K_Δ kernel
K_ij = (z_i / (z_i**2 + ||x_biased_i - x_biased_j||**2 + eps)) ** delta
A_ij = softmax(K_ij, axis=-1)                             # normalized attention
out  = A_ij @ V                                           # (N_tok, d_head)
```

### 4.3 SimpleSSM

Diagonal scan using `jax.lax.scan` (not a Python loop):

```python
def step(h, x_t):
    h_new = jnp.exp(self.A) * h + self.B @ x_t
    y_t   = self.C @ h_new + self.D * x_t
    return h_new, y_t

h0 = jnp.zeros(self.d_state)
_, ys = jax.lax.scan(step, h0, xs)    # xs: (seq_len, d_model)
```

### 4.4 HALOBackbone

8-layer list `[SimpleSSM, SimpleSSM, SimpleSSM, HoloAttention, SimpleSSM, SimpleSSM, SimpleSSM, HoloAttention]`. Layer type encoded as static tuple `("S","S","S","H","S","S","S","H")` in config — no Python branching inside JIT. Each layer: LayerNorm → core → residual → LayerNorm → FFN → residual.

`delta_x` (action bias) is passed only to HoloAttention layers. `delta_v` (belief conditioning) is added to the flow prior outside the backbone.

### 4.5 AdS-KG Prior

Pure JAX function (no parameters):

```python
def ads_kg_prior(x_noise, x_data, t, delta_flow, delta_v):
    z_t     = 1.0 - t
    target_v = x_data - x_noise
    K       = (z_t / (z_t**2 + jnp.sum((x_data - x_noise)**2, axis=-1) + eps)) ** delta_flow
    v_kg    = jax.nn.softmax(K) @ target_v
    return v_kg + delta_v                # belief-conditioned flow
```

### 4.6 PageCurveMemory

Mutable Python list replaced by a fixed-size JAX ring buffer:

```python
# State: cache: Array[max_cache, d_model], ptr: int, island: Array[island_size, d_model]
# Add token: scatter-write at ptr % max_cache
# Eviction: jnp.argsort(S_gen) + jnp.where masking (JIT-safe, no Python indexing)
```

**Port validation:** `test_halo_jax.py` copies weights from the PyTorch reference via numpy and checks that stateless components (HoloEmbedding, HoloAttention, AdS-KG) produce numerically matching outputs (atol=1e-4).

---

## 5. Bridge Interface (`halo_fep/bridge/`)

All three bridges are `eqx.Module` with learned linear weights. Gradients flow in both directions through each bridge.

### 5.1 ObsBridge (`obs_bridge.py`)

Maps HALO backbone output to per-agent discrete observations.

```python
# h_out: (N_tok, d_model)
# Learned token assignment: (N_agents, N_tok), softmax-normalized
h_pooled = assignment @ h_out                  # (N_agents, d_model)
logits   = W_obs @ h_pooled.T                  # (n_obs, N_agents)
s_i      = jnp.argmax(logits, axis=0)          # (N_agents,)
```

`assignment: (N_agents, N_tok)` — each agent learns which tokens to attend to. Initialized uniform `1/N_tok`, row-wise softmax applied at each forward pass to keep weights normalized. `W_obs: (n_obs, d_model)`.

### 5.2 ActionBridge (`action_bridge.py`)

Maps agent actions to K_Δ boundary bias. Agents steer holographic attention by shifting boundary token positions.

```python
# a_i: (N_agents, n_actions)
agent_x_bias = W_action @ a_i.T               # (d_boundary, N_agents)
delta_x      = (agent_x_bias @ assignment).T  # (N_tok, d_boundary)
```

`W_action: (d_boundary, n_actions)`.

### 5.3 BeliefBridge (`belief_bridge.py`)

Maps agent beliefs to AdS-KG flow conditioning. Agents push the generative flow toward observations consistent with their beliefs.

```python
# mu_i: (N_agents, n_hidden)
agent_v_bias = W_belief @ mu_i.T              # (d_model, N_agents)
delta_v      = (agent_v_bias @ assignment).T  # (N_tok, d_model)
```

`W_belief: (d_model, n_hidden)`.

### 5.4 Shapes Summary

| Bridge | Input | Output | Parameters |
|---|---|---|---|
| ObsBridge | `(N_tok, d_model)` | `(N_agents, n_obs)` | `assignment (N_agents, N_tok)`, `W_obs (n_obs, d_model)` |
| ActionBridge | `(N_agents, n_actions)` | `(N_tok, d_boundary)` | `W_action (d_boundary, n_actions)` |
| BeliefBridge | `(N_agents, n_hidden)` | `(N_tok, d_model)` | `W_belief (d_model, n_hidden)` |

---

## 6. Unified Model + Training

### 6.1 HaloFEPModel (`model.py`)

Single `eqx.Module` containing all sub-modules. Closed-loop step is `eqx.filter_jit`-compiled:

```python
@eqx.filter_jit
def halo_fep_step(model, carry, multimodal_tokens):
    swarm_state, page_mem, gm_params = carry

    # 1. FEP feedback → HALO conditioning (from previous step)
    delta_x = model.action_bridge(swarm_state.action)   # K_Δ boundary bias
    delta_v = model.belief_bridge(swarm_state.mu)       # flow conditioning

    # 2. HALO forward
    h_out, page_mem = model.backbone(multimodal_tokens, delta_x, delta_v, page_mem)

    # 3. HALO → FEP observations
    obs = model.obs_bridge(h_out)                       # (N_agents, n_obs)

    # 4. FEP swarm step (imported unchanged from fep_swarm)
    swarm_state = swarm_step(swarm_state, obs, gm_params, cfg)

    return (swarm_state, page_mem, gm_params), (h_out, obs, swarm_state)
```

Compatible with `jax.lax.scan` for JIT-compiled multi-step rollouts.

### 6.2 Joint Loss (`loss.py`)

```python
def halo_fep_loss(model, batch, carry, cfg):
    carry_new, (h_out, obs, swarm_state) = halo_fep_step(model, carry, batch["tokens"])

    # HALO loss
    L_halo = halo_loss(h_out, batch["v_target"], cfg)   # L_FM + L_Bek + L_thermo + L_page

    # FEP free energy (mean over agents)
    F_swarm = jnp.mean(
        jax.vmap(free_energy)(swarm_state.mu, obs, gm_params)
    )

    L_total = L_halo + cfg.lambda_fep * F_swarm
    return L_total, carry_new
```

### 6.3 Training Loop

```python
params, static = eqx.partition(model, eqx.is_array)
opt = optax.adam(cfg.lr)
opt_state = opt.init(params)

@jax.jit
def train_step(params, static, opt_state, carry, batch):
    model = eqx.combine(params, static)
    (loss, carry_new), grads = jax.value_and_grad(
        halo_fep_loss, has_aux=True
    )(model, batch, carry, cfg)
    updates, opt_state_new = opt.update(grads, opt_state, params)
    params_new = eqx.apply_updates(params, updates)
    return params_new, opt_state_new, carry_new, loss
```

Single `optax.adam(cfg.lr)` over all parameters. `lambda_fep=0.1` balances HALO and FEP objectives.

---

## 7. Benchmark Task

### 7.1 Multimodal Goal Inference (`benchmark/multimodal_world.py`)

```
Hidden goal η ∈ {0, ..., n_hidden-1}   (8 goal classes)

Per step:
  text_embed  ~ N(μ_text[η],  0.1·I)   # 768-dim, weak noise
  image_embed ~ N(μ_image[η], 0.3·I)   # 768-dim, stronger noise

μ_text[η], μ_image[η]: fixed random centers drawn at world init
η: fixed per episode, randomized across episodes
```

HALO processes `(text_embed, image_embed)` → `h_out`. ObsBridge maps `h_out` → agent observations `s_i`. Agents infer `η` via belief update over 50 steps.

**Success:** episode succeeds when `argmax(μ_i) == η` for ≥80% of agents at step 50.

### 7.2 Evaluation (`benchmark/eval.py`)

```python
def run_episode(model, cfg, key):
    eta = jax.random.randint(key, (), 0, cfg.n_hidden)
    carry = init_carry(model, cfg, key)
    for t in range(50):
        tokens = sample_tokens(eta, cfg, key)
        carry, state = halo_fep_step(model, carry, tokens)
    mu_final  = state.swarm_state.mu               # (N_agents, n_hidden)
    predicted = jnp.argmax(mu_final, axis=-1)      # (N_agents,)
    return jnp.mean(predicted == eta) >= 0.8

def benchmark(model, cfg, n_episodes=100):
    keys = jax.random.split(jax.random.PRNGKey(0), n_episodes)
    successes = jax.vmap(run_episode, in_axes=(None, None, 0))(model, cfg, keys)
    return jnp.mean(successes)                     # target: ≥ 0.80
```

---

## 8. Test Plan

### test_halo_jax.py — Port Correctness

- `test_holo_embedding_shape`: output `(N_tok, d_boundary)` and `(N_tok, 1)`, no NaN
- `test_holo_attention_shape`: output `(N_tok, d_model)`, no NaN
- `test_holo_attention_matches_pytorch`: numerically matches PyTorch reference (atol=1e-4, weights copied via numpy)
- `test_simple_ssm_shape`: output `(seq_len, d_model)`, no NaN
- `test_simple_ssm_scan_vs_loop`: `jax.lax.scan` output matches manual Python loop (atol=1e-6)
- `test_ads_kg_prior_shape`: output shape `(N_tok, d_model)`, no NaN
- `test_page_memory_evicts_correctly`: cache never exceeds `max_cache` after 200 tokens
- `test_backbone_shape`: `h_out` shape `(N_tok, d_model)`, JIT-compiles without error
- `test_backbone_action_bias_changes_output`: `h_out` differs with nonzero `delta_x`
- `test_backbone_belief_conditioning_changes_output`: `h_out` differs with nonzero `delta_v`

### test_bridge.py — Bridge Interface

- `test_obs_bridge_shape`: output `(N_agents, n_obs)`, no NaN
- `test_action_bridge_shape`: output `(N_tok, d_boundary)`, no NaN
- `test_belief_bridge_shape`: output `(N_tok, d_model)`, no NaN
- `test_obs_bridge_gradients_flow`: `jax.grad` through obs_bridge returns non-zero grads
- `test_action_bridge_gradients_flow`: grads flow back to action inputs
- `test_belief_bridge_gradients_flow`: grads flow back to belief inputs

### test_model.py — Unified Model

- `test_closed_loop_step_shape`: all outputs correct shapes, no NaN/Inf
- `test_closed_loop_jit_compiles`: `eqx.filter_jit(halo_fep_step)` compiles without error
- `test_joint_loss_scalar`: `halo_fep_loss` returns scalar, no NaN
- `test_joint_loss_gradients_nonzero`: `jax.grad` returns nonzero grads for all parameter groups
- `test_train_step_decreases_loss`: loss at step 10 < loss at step 0

### test_benchmark.py — Behavioral Proof

- `test_episode_no_error`: 50-step episode runs without NaN/shape error
- `test_belief_entropy_decreases`: `H(μ)` at step 50 < `H(μ)` at step 0 (agents become certain)
- `test_action_modulates_halo`: `h_out` differs with vs without action bridge (`delta_x` nonzero)
- `test_belief_modulates_flow`: `v_KG` differs with vs without belief bridge (`delta_v` nonzero)
- `test_benchmark_success_rate` *(slow)*: trained model achieves ≥80% success over 100 episodes

---

## 9. What is NOT in Scope

- Real image/text data — synthetic embeddings only
- Multi-GPU training — single device prototype
- Video/audio modalities — text + image only
- Macro coarse-graining on the integrated system — FEP-Swarm macro layer runs independently
- End-to-end backprop through the FEP discrete A/B/C/D parameters — gm_params are treated as fixed during joint training in v1; joint GM training is a future extension
- Replacing the existing `fep_swarm/` or `halo/` modules — both kept intact

---

## 10. Success Criteria

1. All tests in `halo_fep/tests/` pass: `pytest halo_fep/tests/ -v -m "not slow"`
2. Stateless HALO-JAX components match PyTorch numerically (atol=1e-4)
3. Closed-loop step JIT-compiles and runs 50 steps on CPU without OOM (batch=1)
4. `test_action_modulates_halo` and `test_belief_modulates_flow` both pass (closed loop is active)
5. *(slow)* Benchmark success rate ≥ 80% over 100 episodes after 10k training steps
