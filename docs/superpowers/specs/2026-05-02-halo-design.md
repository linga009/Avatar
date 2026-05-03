# HALO: Holographic AdS-Learned Omnimodal Architecture
## Design Specification

**Author:** Design session — May 2, 2026
**Status:** Approved for implementation

---

## 1. Goal

Build a novel multimodal AI research prototype whose architecture is derived from the mathematics of Anti-de Sitter/Conformal Field Theory (AdS/CFT) holography. HALO fuses text and image modalities using a hyperbolic attention kernel grounded in the AdS bulk-to-boundary propagator, an AdS Klein-Gordon generative flow prior, and a Page-curve-driven memory manager. Every architectural choice has a physics derivation, not an empirical one.

---

## 2. Motivation and Theoretical Basis

### 2.1 The Central Correspondence

A transformer is a discretized AdS spacetime:

| AdS/CFT | HALO |
|---|---|
| Bulk radial coordinate z | Transformer layer depth |
| Boundary (z→0) | Final-layer representations |
| CFT operators with dimension Δ | Modality tokens with learned Δ |
| Bulk-to-boundary propagator K_Δ | Attention kernel |
| Klein-Gordon field in AdS | Generative flow prior |
| Page curve / island formula | KV-cache eviction policy |
| Holographic RG flow | Layer-by-layer coarse-graining |

### 2.2 Four Key Physics Imports

**A. HoloAttention (from AdS/CFT §7 of the PDF)**

The bulk-to-boundary propagator for a scalar of conformal dimension Δ in AdS_{d+1}:

```
K_Δ(z, x; x') = C_Δ · (z / (z² + |x - x'|²))^Δ
```

Used as the attention kernel: token i at depth z_i attends to token j at position x_j via K_Δ. Conformal dimension Δ is a learnable parameter per attention head, initialized by modality type.

**B. AdS-KG Flow Prior (from Bogoliubov analysis §3.3 of the PDF)**

The Klein-Gordon equation in AdS provides an analytical vector field for flow matching. The flow prior is:

```
v_t^KG(x_q) = Σ_k K_Δ(z_t, x_q; x_k) · (x_k^data - x_k^noise)
               ─────────────────────────────────────────────────
               Σ_k K_Δ(z_t, x_q; x_k)
```

where z_t = 1 - t (depth decreases toward boundary as generation progresses). The neural net learns only the residual correction ε_θ.

**C. Page Curve Memory (from §8 of the PDF — island formula)**

For a KV-cache of N tokens, the Page time is t_P = N·d_head/(2·d_model). After t_P, evict the token minimizing generalized entropy:

```
S_gen(i) = ||x_i||² · d_head / 4  +  H(a_i)
           ────────────────────────   ────────
           area term (Bekenstein)     von Neumann entropy of attention row
```

Evicted tokens are compressed into an "island" episodic buffer.

**D. Bekenstein Regularizer (from §3.4 of the PDF)**

Attention entropy at each layer is bounded by the holographic bound:

```
L_Bek^(l) = max(0,  H(A^(l)) - α · N · d_head)
```

This penalizes attention entropy exceeding the area-law bound, replacing dropout.

---

## 3. Architecture

### 3.1 File Structure

```
halo/
├── config.py                    # HaloConfig dataclass — all hyperparameters
├── embeddings/
│   ├── holo_embedding.py        # Poincaré half-space lifting (x, z)
│   └── modality_encoders.py     # TextEncoder, ImageEncoder
├── attention/
│   └── holo_attention.py        # K_Δ kernel + HoloAttention module
├── backbone/
│   ├── simple_ssm.py            # Diagonal-state linear SSM layer
│   └── halo_backbone.py         # Alternating SSM + HoloAttention
├── flow/
│   └── ads_kg_prior.py          # AdS Klein-Gordon flow prior
├── memory/
│   └── page_curve_memory.py     # Island-formula KV eviction
├── heads/
│   ├── text_head.py             # Δ=1 text output head
│   └── image_head.py            # Δ=2 image latent head
├── loss.py                      # HALOLoss: FM + Bekenstein + thermo + page
├── model.py                     # HALOModel (pl.LightningModule)
├── data/
│   ├── synthetic_dataset.py     # Paired (text-embedding, image-embedding) synthetic data
│   └── collate.py               # halo_collate batch function
└── tests/
    ├── test_holo_embedding.py
    ├── test_holo_attention.py
    ├── test_ads_kg_prior.py
    ├── test_page_curve_memory.py
    ├── test_backbone.py
    ├── test_heads.py
    ├── test_loss.py
    ├── test_model.py
    └── test_integration.py
```

### 3.2 HaloConfig

```python
@dataclass
class HaloConfig:
    # Dimensions
    d_model: int = 256        # Main model dimension
    d_boundary: int = 64      # Poincaré boundary coordinate dimension
    d_head: int = 64          # Attention head dimension
    n_heads: int = 4          # Number of HoloAttention heads

    # Backbone
    n_layers: int = 8         # Total layers (6 SSM + 2 HoloAttention, pattern [S,S,S,H,S,S,S,H])
    d_state: int = 16         # SSM state dimension
    d_ff: int = 512           # Feed-forward dimension

    # Modality conformal dimensions (learnable, initialized here)
    delta_text: float = 1.0
    delta_image: float = 2.0

    # Flow matching
    flow_steps: int = 4       # Euler ODE steps at inference
    delta_flow: float = 1.5   # KG prior conformal dimension

    # Memory
    max_cache: int = 128      # Max tokens in active KV cache
    island_size: int = 32     # Episodic island buffer size
    bekenstein_alpha: float = 0.1

    # Encoders
    clip_model: str = "ViT-L/14"
    clip_pretrained: str = "openai"
    vocab_size: int = 50257   # GPT-2 tokenizer
    text_embed_dim: int = 768 # CLIP text embedding dim
    image_embed_dim: int = 768 # CLIP image embedding dim

    # Training
    lr: float = 3e-4
    fisher_lambda: float = 1e-3  # Fisher regularization for GEO optimizer
    lambda_bek: float = 0.1
    lambda_thermo: float = 0.05
    lambda_page: float = 0.05
```

### 3.3 HoloEmbedding

Takes a token embedding h ∈ R^{d_model} and projects it into Poincaré half-space:
- x = W_x · h ∈ R^{d_boundary}  (boundary position)
- z = σ(W_z · h) ∈ (0, 1)      (depth; 1=deep/UV, 0=shallow/IR)

### 3.4 HoloAttention

Attention weight from token i to token j:

```
A_ij = K_Δ(z_i, x_i; x_j) / Σ_k K_Δ(z_i, x_i; x_k)

K_Δ(z_i, x_i; x_j) = (z_i / (z_i² + ||x_i - x_j||² + ε))^Δ
```

Δ is a per-head learnable parameter stored as log_delta (to enforce Δ > 0):
`delta = exp(log_delta)`, initialized to `log(delta_init)`.

Value projection: standard linear layer on d_model.
Output: weighted sum of value projections, projected back to d_model.

### 3.5 SimpleSSM

Diagonal-state linear SSM (simplified Mamba without selective scan):
```
h_t = exp(A) ⊙ h_{t-1} + B·x_t
y_t = C·h_t + D⊙x_t
```
where A is a learnable diagonal state matrix (d_state,), B: d_model→d_state, C: d_state→d_model, D: skip scalar vector.

Sequential scan across the sequence dimension.

### 3.6 HALOBackbone

8-layer stack with pattern [SSM, SSM, SSM, HoloAttn, SSM, SSM, SSM, HoloAttn].
Each layer: LayerNorm → core layer → residual → LayerNorm → FFN → residual.

### 3.7 AdS-KG Prior

Given a batch of (x_noise, x_data) pairs, computes the AdS-KG-weighted flow vector field at time t:
1. z_t = 1 - t  (depth at time t)
2. target_v = x_data - x_noise  (straight-line OT target)
3. K = K_Δ(z_t, x_query; x_noise)  (AdS kernel weights)
4. v_KG = softmax(K) @ target_v  (weighted average vector field)

Returns v_KG as the prior; neural net learns residual ε_θ.

### 3.8 PageCurveMemory

State: active_cache (list of token KV pairs), island_buffer (compressed).

Protocol:
1. Add new token to active_cache.
2. If len(active_cache) ≤ max_cache: do nothing (before Page time behavior).
3. If len(active_cache) > max_cache:
   - Compute S_gen(i) = ||x_i||² · d_head/4 + H(a_i) for each cached token.
   - Evict argmin(S_gen) into island_buffer (FIFO, max island_size).
   - Return attention over active_cache + island_buffer (island tokens attend read-only).

### 3.9 Modality Encoders

**TextEncoder:**
- CLIP tokenizer → token ids → open_clip text model (frozen) → 768-dim embedding
- Linear probe: 768 → d_model
- Conformal dimension: delta_text (learnable scalar, init 1.0)

**ImageEncoder:**
- open_clip ViT-L/14 (frozen) → 768-dim embedding
- Linear probe: 768 → d_model
- Conformal dimension: delta_image (learnable scalar, init 2.0)

### 3.10 Output Heads

**TextHead (Δ=1):**
- Linear: d_model → vocab_size
- Uses delta_text to scale logits: logits / delta_text^0.5

**ImageHead (Δ=2):**
- Linear: d_model → image_embed_dim (768)
- Projects to CLIP image embedding space for reconstruction loss

### 3.11 HALOLoss

```
L_total = L_FM + λ_bek · L_Bek + λ_thermo · L_thermo + λ_page · L_page

L_FM    = MSE(v_pred, v_target)         # flow matching loss
L_Bek   = mean(max(0, H(A^l) - α·N·d_head))  # Bekenstein regularizer
L_thermo = max(0, ε_prod_min - ε_prod)  # entropy production lower bound
L_page  = KL(evict_dist || S_gen_dist)  # page curve alignment
```

For entropy production: ε_prod = mean(||v_pred||² / 2), minimum bound = 0.01.

### 3.12 HALOModel (pl.LightningModule)

Forward pass (training, flow matching mode):
1. Encode text → (h_text, Δ_text)
2. Encode image → (h_image, Δ_image)
3. Concatenate: h = [h_text; h_image], shape (B, N_text + N_image, d_model)
4. HoloEmbedding: (x, z) ← holo_embed(h)
5. Sample t ~ Uniform(0, 1), x_noise ~ N(0, I)
6. x_t = (1-t)·x_noise + t·x_data (interpolation in embedding space)
7. Compute v_KG = ads_kg_prior(x_t, x_noise, x_data, t)
8. Run backbone: h_out = backbone(x_t, x, z, page_memory)
9. v_pred = (text_head(h_out[:N_text]) via text path) + (image_head(h_out[N_text:]) via image path)
10. Compute L_total
11. Log all loss components

Inference (generation):
- 4-step Euler ODE: x_{t+dt} = x_t + dt · (v_KG(x_t, t) + ε_θ(x_t, t))
- Decode x_1 through output heads

---

## 4. Data

### Synthetic Dataset (for architecture validation)

Generate N paired (text-embedding, image-embedding) samples:
- text_embed ~ N(μ_t, Σ_t) with μ_t sampled from a mixture of 5 Gaussians
- image_embed = A · text_embed + noise, where A is a fixed random rotation matrix

This creates a structured cross-modal correlation that HALO should learn. A flat Euclidean flow model has no physics prior; HALO's K_Δ kernel should converge faster.

The validation benchmark: plot training loss (L_FM) vs. steps for HALO vs. baseline (same architecture, dot-product attention, Euclidean flow). Expect HALO to converge in fewer steps (validated by GenAdS paper).

---

## 5. What is NOT in Scope

- Real dataset loading (LAION, CC3M, etc.) — synthetic data only for prototype
- Multi-GPU training — single GPU / CPU prototype
- Video or audio modalities — text + image only
- Full Mamba (selective scan) — SimpleSSM is sufficient for architectural validation
- Production deployment — research prototype only

---

## 6. Success Criteria

1. All tests pass: `pytest halo/tests/ -v` → ≥ 95% pass rate
2. Forward pass runs on CPU without OOM: batch_size=4, N_text=32, N_image=16
3. HoloAttention produces different patterns per modality (Δ_text ≠ Δ_image after training)
4. Page curve memory evicts tokens correctly: cache never exceeds max_cache
5. HALO L_FM converges faster than Euclidean baseline on synthetic dataset (validated by learning curve plot)
6. Bekenstein regularizer reduces attention entropy below holographic bound
