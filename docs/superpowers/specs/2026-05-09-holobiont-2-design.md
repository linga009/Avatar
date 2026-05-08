# HoloBiont 2.0 "Holographic Oracle" — Design Spec

> Physics-native redesign optimized for 6 GB VRAM (GTX 1660 Ti).
> Replaces HoloBiont v1 (halo_fep) with a ground-up architecture
> incorporating findings from 7 parallel research agents, 80+ papers.

---

## Goal

Build the highest-fidelity persistent mind that trains on a single 6 GB GPU.
Every architectural choice is physics-motivated — no ad-hoc engineering compromises.
Training VRAM budget: **1.6 GB** (4.4 GB headroom for batch/sequence scaling).

## Architecture Overview

A 24-layer hybrid SSM-attention backbone on a Lorentz hyperboloid, with a mean-field
Active Inference swarm of 32 cluster centroids, latent flow matching in 64-dim boundary
space, and shortcut conditioning for 1-NFE inference.

```
Perception (web + embed)
    │
    ▼
LorentzEmbedding → (x ∈ Lorentz^64, z derived from x_0)
    │
    ▼
Halo2Backbone: [S,S,S,S,S,H] × 4 = 24 layers
  ├─ 20× SelectiveSSM (S7, d_state=64, parallel scan)
  ├─ 4× SharedHoloAttention (2 shared blocks, Zamba2 + LoRA r=16)
  ├─ 24× SwiGLU FFN (d_ff=2816)
  └─ All layers: jax.checkpoint + FP16
    │
    ▼
Latent Flow Matching (boundary space, d=64)
  ├─ ads_kg_prior on Lorentz manifold
  ├─ Shortcut conditioning (step_size scalar)
  └─ Riemannian geodesic interpolation
    │
    ▼
Mean-Field FEP Swarm (K=32 clusters)
  ├─ ObsBridge: backbone → (32, n_obs) observations
  ├─ belief_update_mp: 1-step message passing (no gradients)
  ├─ dpefe_action_selection: backward Bellman DP
  ├─ ActionBridge: (32, n_actions) → boundary bias
  └─ BeliefBridge: (32, n_hidden) → flow conditioning
    │
    ▼
MetaLayer (every K=10 ticks) + HomeostaticRegulator (every tick)
    │
    ▼
EpisodeStore (SQLite + FAISS) → Nightly Dreaming (Halo2Trainer)
```

---

## 1. Backbone — "The Bulk"

### 1.1 Dimensions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_model | 1024 | Sweet spot: wide enough for capacity, narrow enough for 24 layers |
| d_head | 128 | Standard for quality attention |
| n_heads | 8 | d_model / d_head |
| n_layers | 24 | Deep > wide at sub-billion scale (MobileLLM, ICML 2024) |
| d_ff | 2816 | ~2.75× d_model, SwiGLU effective expansion |
| d_state | 64 | Richer recurrent memory than v1's 32 |
| d_boundary | 64 | Lorentz coordinates: 1 time + 63 space |
| layer_pattern | SSSSSH × 4 | 20 SSM + 4 attention positions |

### 1.2 SelectiveSSM (S7)

Replaces `SimpleSSM`. Adds input-dependent gating for selective memory.

```
Input x_t: (d_model,)
  gate = sigmoid(W_gate · x_t)          # (d_state,) — what to remember
  dt   = softplus(W_dt · x_t)           # (d_state,) — time step
  B_t  = W_B · x_t                      # (d_state,) — input projection
  C_t  = W_C · x_t                      # (d_state,) — output projection

  h_t  = gate ⊙ (exp(A · dt) ⊙ h_{t-1}) + (1 - gate) ⊙ (B_t · x_t)
  y_t  = C_t · h_t + D ⊙ x_t
```

- **Scan**: `jax.lax.associative_scan` for parallel training
- **Params per layer**: W_gate(1024×64) + W_dt(1024×64) + W_B(1024×64) + W_C(64×1024) + A(64) + D(1024) = ~263K
- **Total SSM params**: 20 × 263K = 5.3M

### 1.3 SharedHoloAttention (Zamba2-style)

Two shared attention blocks (block_A, block_B) with per-position LoRA adapters.

```
Positions: [5, 11, 17, 23]
  Position 5:  block_A(h, x, z) + lora_A_0
  Position 11: block_B(h, x, z) + lora_B_0
  Position 17: block_A(h, x, z) + lora_A_1
  Position 23: block_B(h, x, z) + lora_B_1
```

Each shared block contains:
- `log_delta`: (n_heads,) learnable conformal dimensions
- `v_proj`: Linear(d_model, n_heads × d_head)
- `out_proj`: Linear(n_heads × d_head, d_model)

Each LoRA adapter:
- `lora_A_v`: (d_model, 16) on v_proj
- `lora_B_v`: (16, n_heads × d_head) on v_proj
- `lora_A_out`: (n_heads × d_head, 16) on out_proj
- `lora_B_out`: (16, d_model) on out_proj

AdS kernel (Lorentz-native):
```
d_ij = arccosh(-κ · ⟨x_i, x_j⟩_L)           # Lorentz geodesic distance
K_ij = (1 / (cosh(d_ij) + 1))^{exp(log_delta)} # K_Delta kernel
A_ij = softmax_j(K_ij)                         # attention weights
```

- **Shared params**: 2 × (1024² + 1024² + 8) = 4.2M
- **LoRA params**: 4 × (1024×16 + 16×1024) × 2 = 262K
- **Total attention params**: 4.5M

### 1.4 SwiGLU FFN

```
gate = W_gate · x                # (d_model,) → (d_ff,)
up   = W_up · x                  # (d_model,) → (d_ff,)
down = W_down · (SiLU(gate) ⊙ up) # (d_ff,) → (d_model,)
```

- **Params per layer**: 3 × 1024 × 2816 = 8.65M
- **Total FFN params**: 24 × 8.65M = 207.6M

### 1.5 LayerNorm

Pre-norm at each layer (before SSM/attention, before FFN).
- 2 × 24 × 1024 × 2 = 98K params (negligible)

### 1.6 Per-Layer Conformal Dimension

`log_delta` shape: (n_attn_positions=4, n_heads=8) = 32 values.
Each attention position at each depth gets its own conformal dimension,
creating a proper RG flow from UV (early layers) to IR (deep layers).

### 1.7 Gradient Checkpointing

Every layer in the backbone for-loop wrapped with `jax.checkpoint`:
```python
for i, (layer, norm1, norm2, ffn) in enumerate(layers):
    h = jax.checkpoint(layer_fn)(h, layer, norm1, norm2, ffn, ...)
```

---

## 2. Lorentz Embedding

Replaces Poincaré half-space. Numerically stable (no sigmoid clamping),
closed-form geodesics, validated by ICLR 2026 (Intrinsic Lorentz Neural Network).

### 2.1 LorentzEmbedding Module

```
Input: h (n_tokens, d_model)
Output: x (n_tokens, d_boundary), z (n_tokens,)

x_euclidean = W_x · h                             # (n_tokens, d_boundary - 1)
x_0 = sqrt(1/κ + ||x_euclidean||²)                # time component (Lorentz constraint)
x_lorentz = [x_0, x_euclidean]                     # (n_tokens, d_boundary)
z = softplus(x_0 - 1/√κ)                          # radial depth derived from curvature
```

- Curvature `κ` is a learnable scalar (initialized to 1.0)
- No `sigmoid` clamping — the Lorentz constraint ensures validity
- d_boundary = 64: first component is time, remaining 63 are spatial

### 2.2 Lorentz Operations (pure functions, no parameters)

```python
def lorentz_inner(x, y, kappa):
    """Minkowski inner product: -x_0*y_0 + x_1*y_1 + ... + x_d*y_d"""
    return -x[0]*y[0] + jnp.dot(x[1:], y[1:])

def lorentz_distance(x, y, kappa):
    """Geodesic distance on hyperboloid"""
    return jnp.arccosh(jnp.clip(-kappa * lorentz_inner(x, y, kappa), 1.0, None)) / jnp.sqrt(kappa)

def exp_map(x, v, kappa):
    """Exponential map: tangent vector v at point x → point on hyperboloid"""
    v_norm = jnp.sqrt(jnp.clip(lorentz_inner(v, v, kappa), 1e-8, None))
    return jnp.cosh(v_norm) * x + jnp.sinh(v_norm) * v / v_norm

def log_map(x, y, kappa):
    """Logarithmic map: point y → tangent vector at x"""
    alpha = -kappa * lorentz_inner(x, y, kappa)
    return (jnp.arccosh(jnp.clip(alpha, 1.0, None)) / jnp.sqrt(jnp.clip(alpha**2 - 1, 1e-8, None))) * (y - alpha * x)
```

---

## 3. FEP Swarm — Mean-Field Clusters

### 3.1 SwarmState

```python
class SwarmState(NamedTuple):
    cluster_mu: jnp.ndarray       # (K, n_hidden) — centroid beliefs
    cluster_action: jnp.ndarray   # (K, n_actions) — centroid policies
    cluster_var: jnp.ndarray      # (K, n_hidden) — within-cluster variance
    key: jnp.ndarray
```

K=32 clusters. `n_agents` in config is the logical agent count (for metrics/reporting),
but computation runs on K centroids only.

### 3.2 Message-Passing Belief Update

```python
def belief_update_mp(mu, soft_obs, gm):
    """Exact posterior in one step. No gradients, no loops."""
    log_likelihood = soft_obs @ jnp.log(gm.A + 1e-8)   # (n_hidden,)
    log_prior = jnp.log(gm.D + 1e-8)                    # (n_hidden,)
    return log_prior + log_likelihood
```

Replaces 16-step `jax.grad` through `fori_loop`. Mathematically identical result
(this IS the fixed point of the gradient descent for categorical distributions).

### 3.3 DPEFE Action Selection

```python
def dpefe_action_selection(mu, gm, cfg):
    """Dynamic Programming EFE via backward Bellman recursion."""
    q_eta = jax.nn.softmax(mu)
    p_obs = jax.nn.softmax(gm.log_C)
    H_s = -jnp.sum(gm.A * jnp.log(gm.A + 1e-8), axis=0)

    V = jnp.zeros(cfg.n_hidden)

    def backward_step(V, _):
        def q_value(a):
            q_next = gm.B[:, :, a] @ q_eta
            q_obs = gm.A @ q_next
            pragmatic = -jnp.sum(q_obs * (jnp.log(q_obs + 1e-8) - jnp.log(p_obs + 1e-8)))
            epistemic = jnp.sum(q_next * H_s)
            future = jnp.sum(q_next * V)
            return pragmatic + epistemic + future
        Q = jax.vmap(q_value)(jnp.arange(cfg.n_actions))
        V_new = jax.nn.softmax(-cfg.beta * Q) @ Q
        return V_new, Q

    _, Q_all = jax.lax.scan(backward_step, V, None, length=cfg.tau)
    return jax.nn.softmax(-cfg.beta * Q_all[0])
```

Complexity: O(tau × n_actions × n_hidden) — linear in all dimensions.

### 3.4 Bridges (K=32)

All three bridges use (K=32, n_tokens=32) assignment matrices:

- **ObsBridge**: backbone output → (32, n_obs) cluster observations
- **ActionBridge**: (32, n_actions) cluster policies → (n_tokens, d_boundary) boundary bias
- **BeliefBridge**: (32, n_hidden) cluster beliefs → (n_tokens, d_model) flow conditioning

Assignment logits: (32, 32) per bridge = 3,072 params total.

### 3.5 Swarm Step

```python
def swarm_step(state, obs, gm, cfg):
    """Update all K clusters in parallel."""
    new_mu = jax.vmap(belief_update_mp, in_axes=(0, 0, None))(
        state.cluster_mu, obs, gm
    )
    new_action = jax.vmap(dpefe_action_selection, in_axes=(0, None, None))(
        new_mu, gm, cfg
    )
    # Update within-cluster variance via EMA
    new_var = 0.99 * state.cluster_var + 0.01 * (new_mu - state.cluster_mu)**2
    return SwarmState(
        cluster_mu=new_mu,
        cluster_action=new_action,
        cluster_var=new_var,
        key=jax.random.split(state.key)[0],
    )
```

---

## 4. Flow Matching — Latent Boundary Space

### 4.1 Latent Flow in Boundary Space (d=64)

All flow matching operates in d_boundary=64, not d_model=1024:

```python
# In halo2_step:
x_data, z_data = lorentz_embed(tokens)          # (n_tokens, 64)
x_noise = sample_lorentz_noise(key, x_data)     # (n_tokens, 64)

# Geodesic interpolation on Lorentz manifold
x_t = exp_map(x_noise, t * log_map(x_noise, x_data, kappa), kappa)

# Velocity target: tangent vector at x_t
v_target = log_map(x_t, x_data, kappa)          # (n_tokens, 64)

# KG prior + learned residual
v_kg = ads_kg_prior_lorentz(x_noise, x_data, t, delta_flow, kappa)
delta_v = belief_bridge(cluster_mu)              # (n_tokens, d_model) → project to 64
v_pred = v_kg + project_to_boundary(delta_v)     # (n_tokens, 64)

# Loss in boundary tangent space
L_flow = MSE(v_pred, v_target)
```

### 4.2 AdS-KG Prior (Lorentz-native)

```python
def ads_kg_prior_lorentz(x_noise, x_data, t, delta_flow, kappa):
    z_t = 1.0 - t
    d_geo = jax.vmap(lorentz_distance, in_axes=(0, 0, None))(x_noise, x_data, kappa)
    K = (z_t / (z_t**2 + jnp.sinh(d_geo)**2 + 1e-6)) ** delta_flow
    weights = jax.nn.softmax(K)
    target = jax.vmap(log_map, in_axes=(0, 0, None))(x_noise, x_data, kappa)
    return weights[:, None] * target
```

Uses `sinh(d_geo)` instead of `||x_i - x_j||²` — the proper hyperbolic distance measure.

### 4.3 Shortcut Conditioning

A small MLP embeds the step_size scalar and adds it to the residual stream:

```python
class StepSizeEmbed(eqx.Module):
    mlp: eqx.nn.MLP  # 1 → 64 → d_model

    def __call__(self, step_size: float) -> jnp.ndarray:
        s = jnp.array([step_size])
        return self.mlp(s)  # (d_model,)
```

Added to every layer's input: `h = h + step_embed` (broadcast over tokens).

During training: `step_size ~ Uniform(0, 1)`.
During inference: `step_size = 1.0` for single-step generation.

---

## 5. Training Pipeline

### 5.1 Optimizer

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=3e-4,
    warmup_steps=500, decay_steps=cfg.n_steps, end_value=3e-5
)
opt = optax.adafactor(learning_rate=schedule)
```

### 5.2 Mixed Precision

FP16 for all forward/backward. FP32 for:
- Lorentz exp_map / log_map (numerical precision)
- SSM state accumulation (drift prevention)
- Loss computation, softmax, log operations
- Adafactor update step

### 5.3 Buffer Donation

```python
@eqx.filter_jit(donate="all")
def train_step(model, opt_state, batch, key):
    (loss, aux), grads = eqx.filter_value_and_grad(
        unified_elbo_loss_v2, has_aux=True
    )(model, carry, batch, key)
    updates, opt_state = opt.update(
        eqx.filter(grads, eqx.is_array), opt_state,
        eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux
```

### 5.4 Unified ELBO Loss v2

```
L_total = L_flow + L_obs + L_prior + λ_bek·L_bek + λ_thermo·L_thermo + λ_page·L_page
```

- **L_flow**: MSE(v_pred, v_target) in boundary tangent space (d=64)
- **L_obs**: -mean Σ_k soft_obs_k · log_A · q_eta_k (K=32 clusters)
- **L_prior**: mean KL[q(η_k) || D] over K clusters
- **L_bek**: Bekenstein attention entropy bound
- **L_thermo**: Entropy production lower bound
- **L_page**: Page curve eviction alignment

### 5.5 Curriculum Learning

```python
sorted_eps = sorted(episodes, key=lambda ep: ep.free_energy_delta)
if step < n_steps * 0.3:
    pool = sorted_eps[:len(sorted_eps) // 2]
else:
    pool = sorted_eps
```

### 5.6 Continual Learning

- EWC penalty on backbone + shared attention params
- Revert-on-diverge: if loss_after > loss_before, discard update
- MESU optimizer optional (same as v1)

---

## 6. VRAM Budget

| Component | Memory |
|-----------|--------|
| Params (310M × 2B FP16) | 620 MB |
| Gradients (310M × 2B FP16) | 620 MB |
| Adafactor state | 8 MB |
| Activations (24 layers, checkpointed, FP16) | 25 MB |
| Page memory (128 × 1024 × 2B) | 0.25 MB |
| Swarm state (32 × 24 × 4B) | 3 KB |
| Bridge assignments (3 × 32 × 32 × 4B) | 12 KB |
| JIT overhead | 400 MB |
| **Total** | **~1,675 MB (1.6 GB)** |
| **Headroom** | **4.4 GB free** |

Headroom supports: batch_size=4, n_tokens=128, or both.

---

## 7. Parameter Count Breakdown

| Component | Params | % |
|-----------|--------|---|
| 20× SelectiveSSM (S7) | 5.3M | 1.7% |
| 2× SharedHoloAttention | 4.2M | 1.4% |
| 4× LoRA adapters | 262K | 0.1% |
| 24× SwiGLU FFN | 207.6M | 66.9% |
| 24× LayerNorm (×2) | 98K | 0.0% |
| LorentzEmbedding | 66K | 0.0% |
| StepSizeEmbed | 65K | 0.0% |
| DiscreteGM (A,B,C,D) | 2.3K | 0.0% |
| MetaLayer | 50K | 0.0% |
| Bridges (×3) | 93K | 0.0% |
| PageCurveMemory | 0 | 0.0% |
| v_proj (boundary→d_model) | 65K | 0.0% |
| **Backbone total** | **217.7M** | **70.2%** |
| **FEP + bridges total** | **0.4M** | **0.1%** |
| **Other (embed, meta, etc.)** | **0.2M** | **0.1%** |
| **Grand total** | **~310M** | **100%** |

Note: FFN dominates at 67%. Future optimization path: MERA tensor decomposition
of FFN weights could compress this 10-100× if more capacity is needed elsewhere.

---

## 8. Config Dataclass

```python
@dataclass(frozen=True)
class Halo2Config:
    # Backbone
    d_model: int = 1024
    d_boundary: int = 64
    n_heads: int = 8
    d_head: int = 128
    n_layers: int = 24
    d_state: int = 64
    d_ff: int = 2816
    layer_pattern: str = "SSSSSH"
    n_shared_attn: int = 2
    lora_rank: int = 16

    # Lorentz
    init_curvature: float = 1.0

    # Flow
    delta_flow: float = 1.5
    shortcut: bool = True

    # Page memory
    max_cache: int = 128
    island_size: int = 32

    # Physics loss
    bekenstein_alpha: float = 0.1
    lambda_bek: float = 0.1
    lambda_thermo: float = 0.05
    lambda_page: float = 0.05

    # FEP
    n_hidden: int = 16
    n_obs: int = 8
    n_actions: int = 8
    tau: int = 3
    beta: float = 1.0

    # Swarm (mean-field)
    n_clusters: int = 32
    n_agents: int = 1024   # logical count for metrics
    kappa: float = 0.3

    # Bridge
    n_tokens: int = 32

    # Training
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42
    precision: str = "fp16"

    # Continual learning
    ewc_lambda: float = 0.1

    # Meta-layer
    meta_n_hidden: int = 8
    meta_n_actions: int = 4
    meta_k: int = 10

    # Homeostatic
    homeo_ema_alpha: float = 0.99
    homeo_novelty_threshold_factor: float = 0.8
    homeo_blend_clip: float = 1.0

    # Heartbeat
    tick_interval: int = 60
```

---

## 9. File Map

| File | Responsibility |
|------|---------------|
| `halo2/__init__.py` | Package init |
| `halo2/config.py` | `Halo2Config` dataclass |
| `halo2/lorentz_ops.py` | Pure functions: inner product, distance, exp_map, log_map |
| `halo2/lorentz_embedding.py` | `LorentzEmbedding` module |
| `halo2/ssm_s7.py` | `SelectiveSSM` with parallel scan |
| `halo2/swiglu.py` | `SwiGLU` FFN module |
| `halo2/holo_attention_shared.py` | `SharedHoloAttention` + `LoRAAdapter` |
| `halo2/backbone.py` | `Halo2Backbone` — 24-layer stack with checkpointing |
| `halo2/step_embed.py` | `StepSizeEmbed` for shortcut conditioning |
| `halo2/ads_kg_prior.py` | Lorentz-native KG prior |
| `halo2/page_memory.py` | `PageCurveMemory` (same logic, Lorentz coords) |
| `halo2/model.py` | `Halo2Model`, `Halo2Carry`, `halo2_step` |
| `halo2/loss.py` | `unified_elbo_loss_v2` |
| `halo2/fep/generative_model.py` | `DiscreteGM` (same A,B,C,D structure) |
| `halo2/fep/belief_update.py` | `belief_update_mp` (message passing) |
| `halo2/fep/action_selection.py` | `dpefe_action_selection` (backward Bellman) |
| `halo2/fep/mean_field_swarm.py` | `SwarmState`, `swarm_step` |
| `halo2/bridge/obs_bridge.py` | K=32 cluster ObsBridge |
| `halo2/bridge/action_bridge.py` | K=32 cluster ActionBridge |
| `halo2/bridge/belief_bridge.py` | K=32 cluster BeliefBridge |
| `halo2/intellect/meta_layer.py` | MetaLayer (adapted for cluster swarm) |
| `halo2/intellect/homeostatic_regulator.py` | HomeostaticRegulator (same as v1) |
| `halo2/intellect/goal_updater.py` | GoalUpdater decay-only (same as v1) |
| `halo2/training/trainer.py` | `Halo2Trainer` — Adafactor+FP16+checkpoint+donation+curriculum |
| `halo2/training/bootstrap.py` | Phase 0 pre-training |
| `halo2/memory/episode_store.py` | EpisodeStore (same as v1) |
| `halo2/memory/schema.py` | Episode schema (same as v1) |
| `halo2/main.py` | Heartbeat loop |
| `halo2/tests/test_config.py` | Config validation tests |
| `halo2/tests/test_lorentz.py` | Lorentz ops + embedding tests |
| `halo2/tests/test_ssm.py` | SelectiveSSM tests |
| `halo2/tests/test_attention.py` | SharedHoloAttention + LoRA tests |
| `halo2/tests/test_backbone.py` | Full backbone tests |
| `halo2/tests/test_fep.py` | belief_update_mp + dpefe + swarm tests |
| `halo2/tests/test_flow.py` | Latent flow + KG prior tests |
| `halo2/tests/test_model.py` | End-to-end model tests |
| `halo2/tests/test_loss.py` | Unified ELBO loss tests |
| `halo2/tests/test_training.py` | Trainer integration tests |

---

## 10. Key References

- [Holographic Generative Flows with AdS/CFT](https://arxiv.org/abs/2601.22033) — validates KG flow matching
- [Intrinsic Lorentz Neural Network](https://arxiv.org/abs/2602.23981) — ICLR 2026, Lorentz model
- [Zamba2 Suite](https://arxiv.org/abs/2411.15242) — shared attention + LoRA
- [S7: Selective Simplified SSM](https://arxiv.org/abs/2410.03464) — input-dependent gating
- [MobileLLM](https://arxiv.org/abs/2402.14905) — depth > width, ICML 2024
- [DPEFE](https://arxiv.org/abs/2504.14898) — dynamic programming EFE
- [Message Passing EFE](https://arxiv.org/abs/2508.02197) — exact belief propagation for AIF
- [Riemannian Variational Flow Matching](https://arxiv.org/abs/2502.12981) — curvature-corrected flow
- [Shortcut Models](https://arxiv.org/abs/2410.12557) — 1-NFE via step-size conditioning
- [Resonant Sparse Geometry Networks](https://arxiv.org/abs/2601.18064) — hyperbolic geodesic sparsity
- [Conformal Fields from Neural Networks](https://arxiv.org/abs/2409.12222) — per-layer conformal dims
- [Mean-Field Multi-Agent RL](https://arxiv.org/abs/1802.05438) — population-level MARL
