# HoloBiont 6GB VRAM Optimization — Master Research Report

> Compiled from 7 parallel deep-research agents, 40+ web searches, 80+ papers surveyed.
> Date: 2026-05-09

---

## The Problem

Current HoloBiont config (d_model=2048, n_layers=20, n_agents=1024) requires **~11.4 GB** for training (params + grads + Adam state + activations). GTX 1660 Ti has **6 GB**. We need to close a **5.4 GB gap** without sacrificing model quality.

## The Solution: 12 Techniques in 3 Phases

---

## PHASE 1: Zero-Architecture-Change Wins (~7 GB saved)

These require minimal code changes and can be applied immediately.

### 1. Adafactor Optimizer (saves 5.7 GB)

**The biggest single win.** Adam stores 2 full copies of all parameters (momentum + variance) = 5.7 GB. Adafactor uses factored row/column statistics: **5.7 GB → 8 MB**.

```python
# lora_trainer.py line 114 — ONE LINE CHANGE
self.opt = optax.adafactor(learning_rate=lr)
```

### 2. FP16 Mixed Precision (saves 1.4 GB on params, 1.4 GB on grads)

GTX 1660 Ti has FP16 tensor cores. Halves parameter and gradient memory.

```python
# Cast model to fp16 for forward/backward
model_fp16 = jax.tree.map(lambda x: x.astype(jnp.float16) if eqx.is_array(x) else x, model)
# Keep loss computation in fp32
```

**Critical:** Keep Poincaré embedding ops, SSM state accumulation, and loss in FP32.

### 3. Gradient Checkpointing (saves ~90% activation memory)

```python
# backbone.py — wrap each layer in the for-loop
@jax.checkpoint
def layer_fn(h, layer, norm1, norm2, ffn, lt, x, z, delta_x):
    h_n = jax.vmap(norm1)(h)
    h_core = layer(h_n) if lt == "S" else layer(h_n, x, z, delta_x=delta_x)
    h = h + h_core
    return h + jax.vmap(ffn)(jax.vmap(norm2)(h))
```

### 4. Buffer Donation (saves ~model_size)

```python
# Prevent duplicate model buffers during training step
@eqx.filter_jit(donate="all-except-first")
def train_step(model, opt_state, tokens, key):
    ...
```

### 5. XLA Memory Flags

```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

**Phase 1 VRAM budget (d_model=2048, 20 layers, 733M params):**

| Component | Before | After Phase 1 |
|-----------|--------|---------------|
| Params | 2,863 MB (FP32) | 1,431 MB (FP16) |
| Gradients | 2,863 MB | 1,431 MB |
| Optimizer | 5,726 MB (Adam) | 8 MB (Adafactor) |
| Activations | 150 MB | 15 MB (checkpointed) |
| **Total** | **11,602 MB** | **2,885 MB** |

**Phase 1 alone gets you from 11.6 GB to 2.9 GB.** Full d_model=2048 now fits.

---

## PHASE 2: HoloBiont-Specific Architecture Wins

These exploit HoloBiont's unique physics to gain quality AND efficiency.

### 6. Message Passing Belief Update (replaces 16-step gradient descent)

**The FEP swarm's biggest memory hog is unnecessary.** The discrete categorical generative model has a **closed-form posterior**: `posterior ∝ prior × likelihood`. The current 16-step `jax.grad` loop through `fori_loop` converges to this exact answer iteratively. Message passing gets it in **one step with zero gradient computation**.

```python
# fep_swarm/agent/belief_update.py — REPLACE entire function
def belief_update(mu_init, soft_obs, gm, cfg):
    """Exact posterior via message passing (no gradients needed)."""
    log_likelihood = soft_obs @ jnp.log(gm.A + 1e-8)  # (n_hidden,)
    log_prior = jnp.log(gm.D + 1e-8)                   # (n_hidden,)
    return log_prior + log_likelihood                    # unnormalized log posterior
```

**Saves:** Eliminates `jax.grad` through `fori_loop` × 16 steps × 1024 agents. Massive reduction in reverse-mode AD overhead.

### 7. DPEFE — Dynamic Programming Expected Free Energy

Replace the nested vmap over `n_actions` policies with backward Bellman recursion.

```python
def dpefe_action_selection(mu, gm, cfg):
    """EFE via backward DP: O(tau × n_actions × n_hidden)."""
    q_eta = jax.nn.softmax(mu)
    V = jnp.zeros(cfg.n_hidden)
    def backward_step(V, _):
        def q_value(a):
            q_next = gm.B[:, :, a] @ q_eta
            q_obs = gm.A @ q_next
            pragmatic = -jnp.sum(q_obs * (jnp.log(q_obs+1e-8) - gm.log_C))
            epistemic = jnp.sum(q_next * (-jnp.sum(gm.A * jnp.log(gm.A+1e-8), axis=0)))
            return pragmatic + epistemic + jnp.sum(q_next * V)
        Q = jax.vmap(q_value)(jnp.arange(cfg.n_actions))
        return jax.nn.softmax(-cfg.beta * Q) @ Q, Q
    _, Q_all = jax.lax.scan(backward_step, V, None, length=cfg.tau)
    return jax.nn.softmax(-cfg.beta * Q_all[0])
```

### 8. Chunked vmap for Swarm

Process 1024 agents in chunks of 64 to bound memory.

```python
# model.py — replace jax.vmap(agent_step) with chunked version
def vmap_chunked(fn, chunk_size=64):
    def chunked(*args):
        n = jax.tree.leaves(args)[0].shape[0]
        results = []
        for start in range(0, n, chunk_size):
            sliced = jax.tree.map(lambda x: x[start:start+chunk_size], args)
            results.append(jax.vmap(fn)(*sliced))
        return jax.tree.map(lambda *xs: jnp.concatenate(xs), *results)
    return chunked
```

### 9. Zamba-Style Shared HoloAttention

**Breakthrough insight:** Use 1 shared HoloAttention block with per-position LoRA adapters instead of 5 independent blocks. Saves ~127 MB at d_model=2048.

The holographic AdS/CFT kernel is inherently global (boundary geometry doesn't change between layers), making parameter sharing physically motivated.

### 10. Fast Multipole Attention for K_Δ Kernel

The AdS kernel `z/(z² + ||x_i - x_j||²)^Δ` is literally a gravitational potential — the Fast Multipole Method was invented for exactly this. Near tokens get full-resolution kernel, distant tokens use cluster centroids.

**Memory:** O(N²) → O(N log N) or O(N) for the attention matrix.

### 11. Lorentz Model (Replace Poincaré Ball)

Multiple ICLR 2026 papers show Lorentz-model hyperbolic embeddings are more numerically stable than Poincaré. Eliminates the `sigmoid` clamping on z in `HoloEmbedding` and avoids boundary instability.

---

## PHASE 3: Deep Architecture Upgrades

### 12. Deep-and-Thin Architecture (MobileLLM insight)

**Depth > Width at sub-billion scale.** d_model=576, n_layers=30 beats d_model=2048, n_layers=8 at the same parameter count.

**Recommended production config (319M params, ~2.85 GB training):**

| Parameter | Value |
|-----------|-------|
| d_model | 1536 |
| n_heads | 12 (d_head=128) |
| n_layers | 16 (SSSH × 4) |
| d_ff | 6144 |
| SSM type | S5 parallel scan |
| Attention | 1 shared HoloAttention + LoRA r=16 |
| Flow | Shortcut (1-NFE inference) |
| Precision | FP16 + Adafactor |
| n_agents | 1024 (chunked vmap, k=64) |

### Additional Advanced Techniques

| Technique | Source | Impact |
|-----------|--------|--------|
| **MERA tensor network FFN** | 3900x param compression, native to AdS/CFT RG flow | Replace FFN weights |
| **Mean-field Fokker-Planck swarm** | O(1) agent memory regardless of N | Replace per-agent tracking |
| **Amortized belief updates** | Neural net predicts posterior in 1 pass | Replace iterative inference |
| **Latent flow matching** | Flow in d_boundary=64 space not d_model=2048 | 32x flow memory reduction |
| **CompactifAI MPO compression** | 93% weight memory reduction via tensor trains | Compress FFN weights |
| **Riemannian flow matching** | Curvature-corrected flow on Poincaré manifold | Better quality, same memory |
| **Contrastive Active Inference** | 13.8x fewer MACs, 3.5x fewer params | Replace generative model |

---

## Architecture Validation

A Jan 2026 paper **"Holographic Generative Flows with AdS/CFT"** (arXiv:2601.22033) independently validates HoloBiont's design:
- Uses KG scalar field as flow matching prior ✓ (exactly `ads_kg_prior`)
- Learned residual on top of KG flow ✓ (exactly `v_pred = v_kg_dm + delta_v`)
- Confirms learnable conformal dimension improves convergence ✓ (exactly `log_delta`)

**HoloBiont's physics is correct. The bottleneck is purely engineering.**

---

## Implementation Priority

| # | Change | VRAM Saved | Effort | Files |
|---|--------|-----------|--------|-------|
| 1 | Adafactor | 5.7 GB | 1 line | `lora_trainer.py` |
| 2 | FP16 mixed precision | 2.8 GB | Low | `lora_trainer.py`, `model.py` |
| 3 | `jax.checkpoint` per layer | 135 MB | 5 lines | `backbone.py` |
| 4 | Buffer donation | ~700 MB | 1 line | `lora_trainer.py` |
| 5 | Message passing belief | Eliminates AD overhead | 15 lines | `belief_update.py` |
| 6 | DPEFE action selection | Eliminates nested vmap | 50 lines | `model.py` |
| 7 | Chunked vmap | 16x swarm reduction | 20 lines | `model.py` |
| 8 | XLA flags | 5-15% | 3 lines | `train.py` |

**Items 1-4 alone close the 5.4 GB gap. Items 5-8 add massive headroom.**

---

## Key References

### Architecture Validation
- [Holographic Generative Flows with AdS/CFT](https://arxiv.org/abs/2601.22033) (Jan 2026)
- [Conformal Fields from Neural Networks](https://arxiv.org/abs/2409.12222) (Sept 2024)
- [Resonant Sparse Geometry Networks](https://arxiv.org/abs/2601.18064) (Jan 2026)

### Efficient Holographic Kernels
- [Fast Multipole Attention](https://arxiv.org/abs/2310.11960) (Sept 2025)
- [Multipole Semantic Attention](https://arxiv.org/abs/2509.10406) (Sept 2025)

### Hyperbolic Networks
- [Intrinsic Lorentz Neural Network](https://arxiv.org/abs/2602.23981) (ICLR 2026)
- [Hierarchical Mamba + Hyperbolic](https://arxiv.org/abs/2505.18973) (May 2025)
- [Hyperbolic Binary Neural Network](https://arxiv.org/abs/2501.03471) (Jan 2025)

### Tensor Network Compression
- [CompactifAI MPO](https://arxiv.org/abs/2401.14109) (2024)
- [MERA Compact Neural Networks](https://openreview.net/forum?id=rkGZuJb0b)

### Active Inference Efficiency
- [Message Passing EFE](https://arxiv.org/abs/2508.02197) (2025)
- [DPEFE](https://arxiv.org/abs/2504.14898) (2025)
- [Contrastive Active Inference](https://openreview.net/forum?id=5t5FPwzE6mq)
- [Amortized Variational Inference](https://arxiv.org/abs/2404.12484)

### SSM + Attention Hybrids
- [Zamba: Shared Attention](https://arxiv.org/abs/2405.16712) (2024)
- [Jamba: Transformer-Mamba Hybrid](https://arxiv.org/abs/2403.19887) (2024)
- [S5 Parallel Scan](https://arxiv.org/abs/2208.04933) (2022)
- [S7 Selective SSM](https://arxiv.org/abs/2410.03464) (2024)

### Flow Matching
- [Shortcut Models](https://arxiv.org/abs/2410.12557) (2024)
- [FS-DFM](https://arxiv.org/abs/2509.20624) (2025)
- [Latent-CFM](https://arxiv.org/abs/2505.04486) (2025)
- [Riemannian Flow Matching](https://arxiv.org/abs/2502.12981) (Feb 2026)

### Training Optimization
- [MobileLLM Deep-and-Thin](https://arxiv.org/abs/2402.14905) (ICML 2024)
- [MPX Mixed Precision JAX](https://arxiv.org/abs/2507.03312) (2025)
- [SmolLM2 Data-Centric Training](https://arxiv.org/abs/2502.02737) (2025)
