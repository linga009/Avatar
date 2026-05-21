# HoloBiont 2.1 Scale-Up — RevNets + GaLore + LISA Design Spec

> Scale HoloBiont 2.0 from d_model=1024 (310M params, 1.6 GB) to
> d_model=2048 (1.2B params, ~3.0 GB) by adding three memory-efficient
> training techniques that stack multiplicatively.

---

## Goal

Train a 1.2B-parameter HoloBiont on a single 6 GB GPU by combining
reversible layers (zero activation storage), GaLore (low-rank gradient
projection), and LISA (layerwise importance sampling).

## Architecture Changes

### 1. Reversible Backbone

Split residual stream into `(h1, h2)` each `(n_tokens, d_model/2)`.

```
Forward:
  h1_new = h1 + F(h2)      # F = SSM or Attention
  h2_new = h2 + G(h1_new)  # G = SwiGLU FFN

Reverse (exact, no storage):
  h2 = h2_new - G(h1_new)
  h1 = h1_new - F(h2)
```

Replaces `jax.checkpoint`. Zero activation memory regardless of depth.
Cost: ~2x compute per backward step.

The SSSSSH pattern maps directly:
- S positions: F = SelectiveSSM on half-stream
- H positions: F = SharedHoloAttention on half-stream
- G = SwiGLU FFN always

Internal layer norms, SSM, attention, and FFN all operate on `d_model//2`
width. The full `d_model` is recovered by concatenating `(h1, h2)` at
the output.

### 2. GaLore (Gradient Low-Rank Projection)

For each weight matrix W of shape (m, n) in SwiGLU layers:

```
Every T=200 steps:
  U, S, V = svd(G_full)
  P = U[:, :rank]            # projection matrix

Each step:
  G_proj = P.T @ G           # (rank, n) instead of (m, n)
  update = optimizer(G_proj)
  W -= P @ update             # reconstruct full-rank delta
```

- rank = 128
- Applied to: SwiGLU w_gate, w_up, w_down (67% of all params)
- NOT applied to: SSM projections, attention v_proj/out_proj, LoRA, embedding (small)
- Gradient memory: 2,400 MB → ~400 MB

### 3. LISA (Layerwise Importance Sampled Training)

Each step, randomly select k=2 of 24 backbone layers to update:

```
active_layers = random_choice(24, k=2)
grads = compute_grads(model)  # only for active layers
grads[frozen_layers] = 0
```

Always update (never freeze):
- `lorentz_embed` (66K params)
- All 3 bridges (93K params)
- `gm` (2.3K params)
- `step_embed` (65K params)

Gradient memory for backbone: 2/24 × backbone_grads = ~8% of full.

### Combined VRAM Budget (d_model=2048, 24 layers, 1.2B params)

| Component | Without | With All Three |
|-----------|---------|---------------|
| Params (FP16) | 2,400 MB | 2,400 MB |
| Gradients | 2,400 MB | ~150 MB |
| Optimizer (Adafactor) | 8 MB | 8 MB |
| Activations | 50 MB | 0 MB |
| JIT overhead | 500 MB | 500 MB |
| **Total** | **5,358 MB** | **~3,058 MB** |

---

## Scaled Config

```python
@dataclass(frozen=True)
class Halo2Config:
    # Backbone — scaled up
    d_model: int = 2048
    d_boundary: int = 64
    n_heads: int = 16
    d_head: int = 128
    n_layers: int = 24
    d_state: int = 64
    d_ff: int = 5632
    layer_pattern: str = "SSSSSH"
    n_shared_attn: int = 2
    lora_rank: int = 16

    # RevNet
    reversible: bool = True

    # GaLore
    galore_rank: int = 128
    galore_update_proj_gap: int = 200
    galore_scale: float = 0.25

    # LISA
    lisa_active_layers: int = 2

    # ... rest unchanged from v2.0 ...
```

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `halo2/config.py` | Modify | Add reversible, galore, lisa config fields |
| `halo2/backbone.py` | Modify | Add `ReversibleHalo2Backbone` alongside existing |
| `halo2/training/galore.py` | Create | GaLore optax wrapper with SVD projection |
| `halo2/training/trainer.py` | Modify | Integrate LISA masking + GaLore into training loop |
| `halo2/model.py` | Modify | Use reversible backbone when `cfg.reversible=True` |
| `halo2/tests/test_reversible.py` | Create | Roundtrip reconstruction tests |
| `halo2/tests/test_galore.py` | Create | Low-rank projection correctness tests |
| `halo2/tests/test_scale.py` | Create | d_model=2048 integration smoke test |

---

## Reversible Backbone Detail

### ReversibleHalo2Backbone

```python
class ReversibleHalo2Backbone(eqx.Module):
    """Reversible residual backbone — zero activation storage."""

    # Same components as Halo2Backbone but forward/backward use rev coupling

    def __call__(self, h, x, z, delta_x=None, step_size=1.0):
        # Split into two streams
        d_half = h.shape[-1] // 2
        h1, h2 = h[:, :d_half], h[:, d_half:]

        s_emb = self.step_embed(step_size)
        h1 = h1 + s_emb[None, :d_half]
        h2 = h2 + s_emb[None, d_half:]

        attn_idx = 0
        for i, lt in enumerate(self.layer_types):
            # F: core layer (SSM or attention) on h2
            h2_normed = jax.vmap(self.norms1[i])(h2)
            if lt == "S":
                f_out = self.layers[i](h2_normed)
            else:
                block_id = self.attn_block_ids[attn_idx]
                lora = self.lora_adapters[attn_idx]
                f_out = self.shared_attns[block_id](h2_normed, x, z, delta_x, lora)
                attn_idx += 1
            h1 = h1 + f_out

            # G: FFN on h1
            h1_normed = jax.vmap(self.norms2[i])(h1)
            g_out = jax.vmap(self.ffns[i])(h1_normed)
            h2 = h2 + g_out

        return jnp.concatenate([h1, h2], axis=-1)
```

For backward pass, JAX's custom_vjp reconstructs activations by reversing:
```
h2 = h2_new - G(h1_new)
h1 = h1_new - F(h2)
```

### Note on half-width

SSM, attention, and FFN all operate on `d_model//2 = 1024` internally.
This means each sub-layer has the same compute profile as the current
d_model=1024 backbone — the "doubling" comes from having two parallel
streams, not from doubling each layer's width.

---

## GaLore Detail

### galore_optax wrapper

```python
def galore_wrapper(base_opt, rank, update_proj_gap, scale):
    """Wraps any optax optimizer with GaLore projection."""

    # State: projection matrices P per weight, step counter
    # Every update_proj_gap steps: recompute P via SVD of gradient
    # Each step: project gradient, run base_opt, reconstruct update
```

Applied selectively via a filter: only SwiGLU weight matrices (w_gate,
w_up, w_down) get GaLore projection. Small params use standard optimizer.

---

## LISA Detail

### Training loop integration

```python
def _lisa_gradient_mask(model, key, n_active, n_layers):
    """Zero out gradients for all but n_active random backbone layers."""
    active = jax.random.choice(key, n_layers, shape=(n_active,), replace=False)
    def mask_fn(path, grad):
        # Always keep: lorentz_embed, bridges, gm, step_embed
        # Backbone layers: keep only if index in active set
        for idx in active:
            if f"layers.{idx}" in path or f"ffns.{idx}" in path:
                return grad
        if "backbone" in path:
            return jnp.zeros_like(grad)
        return grad  # non-backbone params always updated
    return jax.tree_util.tree_map_with_path(mask_fn, model, grads)
```

---

## Testing Strategy

### test_reversible.py
- Forward then reverse recovers original (h1, h2) exactly (atol=1e-4)
- Output shape matches non-reversible backbone
- Gradients flow through reversible layers
- JIT compiles

### test_galore.py
- Projected gradient has correct shape (rank, n)
- Reconstructed update has original shape (m, n)
- SVD projection rotates after update_proj_gap steps
- Training with GaLore produces finite loss

### test_scale.py
- d_model=2048 model instantiates
- Single halo2_step produces finite outputs
- Loss is finite and differentiable
- VRAM stays under 4 GB (with all techniques enabled)
