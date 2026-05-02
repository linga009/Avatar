# Multimodal AI Model Design Spec
**Date:** 2026-05-02
**Status:** Approved

---

## Overview

A large-scale multimodal model capable of generating coherent text and images together (interleaved generation). Built on a hybrid Transformer+Mamba backbone (Jamba), a CLIP vision encoder, and a Flow Matching DiT image generation head. Grounded in the continuous-time mathematical framework described in `New AI.txt`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                         │
│  Text tokens ──► Token Embedder                         │
│  Input image ──► CLIP ViT Encoder ──► Linear Projector  │
└────────────────────────┬────────────────────────────────┘
                         │ unified token stream
┌────────────────────────▼────────────────────────────────┐
│              JAMBA HYBRID BACKBONE                      │
│  [Mamba SSM layers] + [Transformer attention layers]    │
│  4-bit NF4 quantized (bitsandbytes QLoRA)               │
│  LoRA adapters: rank 16, alpha 32                       │
│  Base: ai21labs/Jamba-v0.1 or Jamba-1.5-Mini (~1.4B)   │
└────────────────────────┬────────────────────────────────┘
                         │ hidden states h (sequence)
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────┐          ┌──────────────────────────────────┐
│  TEXT HEAD      │          │  IMAGE HEAD                      │
│  LM head        │          │  VAE Encoder/Decoder (frozen)    │
│  (frozen)       │          │  Flow Matching DiT               │
└─────────────────┘          │  Conditioned via Cross-Attention  │
     ▼                       │  on Jamba hidden states h        │
  Text output                └──────────────────────────────────┘
                                              ▼
                              Latent z → ODE solve (4 steps)
                                              ▼
                              VAE Decoder → generated image
```

### Mathematical Grounding
- **Jamba SSM layers:** Implement the HiPPO memory framework — compressing long-context history into orthogonal polynomial coefficients with O(1) inference memory, as derived in the text file
- **Jamba Transformer layers:** Handle global token interactions via the Vlasov-type interacting particle dynamics described in the text
- **VAE:** Encodes pixel space (512×512 or 1024×1024) into a compact latent space (e.g., 64×64×4 for SD v1.5). The DiT operates entirely in this latent space, keeping memory and compute tractable.
- **Flow Matching DiT:** Solves `dx/dt = v_θ(x, t, h)` in the VAE latent space, conditioned on the Jamba hidden state sequence `h` via Cross-Attention layers. Uses straight-line ODE trajectories from the Optimal Transport / Benamou-Brenier formulation. Integrated via 4-step Euler solver at inference.
- **Conditioning via Cross-Attention:** Each DiT block contains a Cross-Attention sublayer where latent patches (queries) attend to the Jamba hidden state sequence (keys/values). This preserves the full sequential context from the backbone rather than collapsing it to a single vector (as AdaGN would require).

---

## Components

| Component | Pretrained Base | Trainable Part |
|---|---|---|
| CLIP ViT Encoder | `openai/clip-vit-large-patch14` | Frozen |
| Image Projector | None (new 2-layer MLP) | Fully trained |
| Jamba Backbone | `ai21labs/Jamba-v0.1` | QLoRA (4-bit NF4, r=16, α=32) |
| VAE | `runwayml/stable-diffusion-v1-5` VAE | Frozen |
| Flow Matching DiT Head | New (DiT-B/2 or DiT-L/2 init) | Fully trained |
| DiT Cross-Attention layers | None (new) | Fully trained |
| Text LM Head | Inherited from Jamba | Frozen |

### Quantization Detail — 4-bit NF4 QLoRA
The Jamba backbone is loaded in **4-bit NF4 quantization** (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`). LoRA adapters are attached on top in full bf16 precision. This reduces backbone VRAM from ~3 GB (bf16) to **~0.7 GB**, freeing substantial headroom for the DiT head and activations.

### VAE Detail
Using the SD v1.5 VAE (`stabilityai/sd-vae-ft-mse`): encodes 512×512 RGB → 64×64×4 latent; decodes back to pixel space. Kept fully frozen throughout training. The Flow Matching DiT operates exclusively on the 64×64×4 latent representation.

---

## Data Flow

### Understanding mode (image + text → text)
```
Image → CLIP → Projector → [visual tokens] ──┐
Text prompt → Embedder → [text tokens]  ──────┴──► Jamba → Text Head → answer
```

### Generation mode (text → text + image)
```
Text prompt → Embedder → Jamba → [IMG] token → hidden states h (sequence)
Gaussian noise z_0 ~ N(0,I) [64×64×4]
z_0 + h → Flow Matching DiT (Cross-Attention on h) → 4-step ODE → z_1
z_1 → VAE Decoder → generated image [512×512×3]
```

### Interleaved mode (text + image → text + image)
```
Mixed inputs → unified token stream → Jamba →
  text tokens  → Text Head
  [IMG] tokens → Flow Matching DiT
```

---

## Training Strategy

### Phase 1 — Projector Warmup
- **Freeze:** Everything except Image Projector
- **Data:** LLaVA-CC3M-Pretrain-595K (~500K image-text pairs)
- **Objective:** Next-token prediction on text conditioned on visual tokens
- **Duration:** ~1–2 hours on A100 40GB

### Phase 2 — Instruction Fine-tuning
- **Trainable:** Image Projector + Jamba LoRA adapters
- **Data:** LLaVA-Instruct-150K + LAION-COCO subset
- **Objective:** Cross-entropy on text tokens + Flow Matching loss on image tokens
- **Duration:** ~4–8 hours on A100 40GB

### Phase 3 — Flow Matching Head Training
- **Trainable:** DiT head (full)
- **Data:** LAION-Aesthetics 6.5+ (~1M pairs)
- **Objective:** `L = E[||v_θ(x_t, t, h) - (x_1 - x_0)||²]`
- **Scheduler:** Cosine with warmup

---

## Memory Budget (A100 40GB)

| Item | VRAM |
|---|---|
| Jamba-Mini 1.4B (4-bit NF4) | ~0.7 GB |
| LoRA adapters (bf16) | ~0.5 GB |
| VAE (frozen, bf16) | ~0.3 GB |
| DiT head + Cross-Attention layers | ~3–5 GB |
| Activations + gradients | ~20–25 GB |
| 8-bit AdamW optimizer states | ~5 GB |
| **Total** | **~30–37 GB ✓** |

**Optimizations:** 4-bit NF4 QLoRA on backbone, bf16 compute dtype, gradient checkpointing, 8-bit AdamW (`bitsandbytes`), `torch.compile`, Google Drive checkpoint mount for Colab resilience.

---

## Error Handling & Safeguards

- Gradient clipping (`max_norm=1.0`) and loss scaling for bf16 stability
- Silent skip of corrupt/oversized batches
- Lightning `ModelCheckpoint` every 500 steps + best-by-val-loss
- Auto-resume from last checkpoint on Colab reconnect (Drive mount)
- `detect_anomaly` enabled during initial debugging runs

---

## Validation Metrics

| Metric | Frequency | Tool |
|---|---|---|
| Text perplexity | Every 1K steps | Built-in |
| Image FID | Every 5K steps | `clean-fid` |
| CLIP alignment score | Every 5K steps | `open_clip` |

---

## Test Plan

| Test | Scope | Checks |
|---|---|---|
| Projector shape | Unit | CLIP output → correct Jamba token dimensions |
| Flow Matching ODE | Unit | Single forward pass → valid image tensor, no NaN |
| LoRA injection | Unit | Jamba params frozen except LoRA; param count matches |
| Understanding mode | Integration | Image + question → coherent text answer |
| Generation mode | Integration | Text prompt → non-degenerate image in ≤4 ODE steps |
| Interleaved mode | Integration | Mixed input → correct routing to text/image heads |

---

## Stack

- **Framework:** PyTorch + PyTorch Lightning
- **Backbone:** Jamba (ai21labs/Jamba-v0.1 or Jamba-1.5-Mini), 4-bit NF4 quantized
- **Vision encoder:** CLIP ViT-L/14 (OpenAI)
- **VAE:** SD v1.5 VAE (`stabilityai/sd-vae-ft-mse`), frozen
- **Image head:** Flow Matching DiT with Cross-Attention conditioning
- **LoRA/QLoRA:** `peft` library (`BitsAndBytesConfig` + `get_peft_model`)
- **Quantization:** `bitsandbytes` (4-bit NF4, bf16 compute)
- **Optimizer:** 8-bit AdamW (`bitsandbytes`)
- **Hardware:** Google Colab A100 40GB
- **Logging:** Weights & Biases or TensorBoard
