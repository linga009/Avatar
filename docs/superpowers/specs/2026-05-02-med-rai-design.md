# MED-RAI: Medical Robotic AI — Design Spec
**Date:** 2026-05-02
**Status:** Approved
**Deployment context:** Research prototype / simulation only (no real hardware)

---

## Overview

MED-RAI adapts the Jamba + Riemannian Flow Matching framework from `New AI.txt` for surgical robotics simulation. The system takes four input modalities (endoscopic video, SE(3) kinematics, force/torque, surgeon speech) and generates three outputs: geodesic SE(3) surgical trajectories, natural language risk alerts/surgical logs, and next-gesture predictions. It is designed to be simulator-agnostic, consuming standardized kinematic and video inputs compatible with SurRoL, AMBF, JIGSAWS, or any ROS-compatible platform.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT ENCODERS                            │
│                                                                  │
│  Endoscopic video ──► Surgical ViT ──────────────────┐          │
│  Kinematics (SE3)  ──► SE(3) Encoder (MLP+exp map) ──┤          │
│  Force / Torque    ──► F/T MLP Encoder ───────────────┤          │
│  Surgeon language  ──► Token Embedder ────────────────┘          │
└──────────────────────────────┬───────────────────────────────────┘
                               │ per-modality token sequences
┌──────────────────────────────▼───────────────────────────────────┐
│              CROSS-MODAL ALIGNMENT LAYER                         │
│  Lightweight cross-attention: modalities attend to each other    │
│  Force spike + video occlusion → fused danger representation     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ unified aligned token stream
┌──────────────────────────────▼───────────────────────────────────┐
│                  JAMBA HYBRID BACKBONE                           │
│  [Mamba SSM layers] — surgical history (30–60 min), O(1) memory  │
│  [Transformer layers] — global cross-token reasoning             │
│  4-bit NF4 QLoRA (r=16, α=32)                                   │
└──────────────────────────────┬───────────────────────────────────┘
                               │ hidden states h
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌──────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│  RFM POLICY HEAD │  │  TEXT HEAD      │  │  GESTURE HEAD        │
│  Riemannian Flow │  │  Risk alerts    │  │  Next-action class.  │
│  Matching on     │  │  Surgical logs  │  │  JIGSAWS vocabulary  │
│  SE(3) manifold  │  │  (LM head)      │  │  (linear classifier) │
└──────────────────┘  └─────────────────┘  └──────────────────────┘
         ▼                    ▼                    ▼
  SE(3) trajectory      Text output          Gesture label
  (geodesic path,       (log / alert)        (grasp/cut/retract…)
   1–2 sec horizon)
```

---

## Mathematical Grounding

- **SE(3) Encoder:** Maps raw end-effector poses $(R, t) \in SE(3)$ into Lie algebra $\mathfrak{se}(3)$ via the logarithmic map before feeding into Jamba — keeping the geometry honest throughout the pipeline
- **Jamba SSM layers:** HiPPO polynomial projection compresses the entire surgical history into O(1) memory, enabling recall of vessel/nerve locations seen 30+ minutes prior
- **Cross-Modal Alignment:** Vlasov-type interacting particle dynamics — modality tokens attract/repel in embedding space before backbone compression, explicitly capturing dangerous cross-modal correlations
- **RFM Policy Head:** Solves the geodesic equation on SE(3) via Flow Matching — the velocity field $v_\theta$ is trained to produce minimum-energy surgical paths. The geodesic equation on the manifold:

$$\frac{d^2x^i}{dt^2} + \Gamma^i_{jk}\frac{dx^j}{dt}\frac{dx^k}{dt} = 0$$

where $\Gamma^i_{jk}$ are the Christoffel symbols of the SE(3) metric. The Flow Matching loss in $\mathfrak{se}(3)$:

$$\mathcal{L}_{RFM} = \mathbb{E}\left[||v_\theta(\xi_t, t, h) - (\xi_1 - \xi_0)||^2\right]$$

- **SE(3) geometry:** Log map $(R,t) \xrightarrow{\log} \xi \in \mathfrak{se}(3)$ on input; exp map $\xi \xrightarrow{\exp} SE(3)$ on output — all generated trajectories stay on the manifold by construction

---

## Components

| Component | Pretrained Base | Trainable Part |
|---|---|---|
| Surgical ViT | `openai/clip-vit-large-patch14` | Frozen + linear probe |
| SE(3) Encoder | None (new MLP + exp/log maps) | Fully trained |
| F/T Encoder | None (new MLP) | Fully trained |
| Language Embedder | Inherited from Jamba tokenizer | Frozen |
| Cross-Modal Alignment | None (new cross-attention block) | Fully trained |
| Jamba Backbone | `ai21labs/Jamba-v0.1` | QLoRA (4-bit NF4, r=16, α=32) |
| RFM Policy Head | None (new DiT-style on SE(3)) | Fully trained |
| Text Head | Inherited from Jamba | Frozen + LoRA |
| Gesture Classifier | None (new linear layer) | Fully trained |

---

## Data Flow

### Inference — trajectory generation
```
Video frames    → Surgical ViT  → v_tokens  ──┐
Kinematic poses → SE(3) Encoder → k_tokens  ──┤
Force/torque    → F/T Encoder   → f_tokens  ──┤─► Cross-Modal Alignment
Surgeon speech  → Tokenizer     → l_tokens  ──┘          │
                                              Unified token stream
                                                       │
                                               Jamba Backbone
                                                       │ h
                              ┌────────────────────────┤
                              ▼                        ▼
                    RFM Policy Head             Text + Gesture Heads
                    Gaussian ξ_0 ~ N(0,I)       ↓            ↓
                    4-step geodesic ODE    Risk alert    Gesture label
                              ↓
                    SE(3) trajectory (next 1–2s)
```

### Danger signal pathway
```
Force spike detected (F/T encoder)
        +
Endoscopic occlusion (ViT low-confidence patch)
        ↓
Cross-Modal Alignment fuses → HIGH-DANGER token
        ↓
Jamba propagates through history context
        ↓
Text Head:    "WARNING: Force anomaly near last known vessel position"
RFM Head:     trajectory curves away from danger region (geodesic reroute)
Gesture Head: predicts RETRACT
```

---

## Training Strategy

### Datasets

| Dataset | Modalities | Use |
|---|---|---|
| JIGSAWS | Kinematics + video, 3 surgical tasks, gesture labels | Primary: trajectory + gesture training |
| AutoLaparo | Endoscopic video, tool tracking | Video encoder fine-tuning |
| RoboTool / EndoVis 2017 | Endoscopic video + instrument segmentation | Visual danger region labeling |
| Synthetic SE(3) trajectories | Generated via SurRoL or random geodesics on SE(3) | RFM Policy Head warmup |
| Surgical report corpus | Text only (PubMed surgical notes) | Text Head continued pretraining |

### Phase 1 — Encoder & Projector Warmup (~1–2 hrs)
- **Freeze:** Jamba backbone and all output heads
- **Train:** Surgical ViT linear probe, SE(3) Encoder, F/T Encoder, Cross-Modal Alignment
- **Objective:** Reconstruction loss on kinematic sequences + contrastive alignment between video and kinematics
- **Data:** JIGSAWS kinematics + AutoLaparo video

### Phase 2 — RFM Policy Head Warmup (~2–3 hrs)
- **Freeze:** Jamba backbone, all encoders
- **Train:** RFM Policy Head only
- **Objective:** $\mathcal{L}_{RFM}$ on synthetic SE(3) geodesic trajectories
- **Data:** Synthetic geodesics + JIGSAWS kinematic demonstrations

### Phase 3 — Joint Fine-tuning (~6–10 hrs)
- **Train:** QLoRA on Jamba + all heads + Cross-Modal Alignment
- **Objective:**

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{RFM} + \lambda_2 \mathcal{L}_{CE}^{text} + \lambda_3 \mathcal{L}_{CE}^{gesture}$$

  Weights: $\lambda_1=1.0$, $\lambda_2=0.5$, $\lambda_3=0.3$
- **Data:** Full JIGSAWS + EndoVis + surgical text corpus
- **Scheduler:** Cosine warmup over first 500 steps

---

## Memory Budget (A100 40GB)

| Item | VRAM |
|---|---|
| Jamba-Mini 1.4B (4-bit NF4) | ~0.7 GB |
| QLoRA adapters (bf16) | ~0.5 GB |
| Surgical ViT (frozen) | ~0.9 GB |
| SE(3) + F/T Encoders | ~0.2 GB |
| Cross-Modal Alignment | ~0.3 GB |
| RFM Policy Head (DiT-style) | ~3–4 GB |
| Activations + gradients | ~20–25 GB |
| 8-bit AdamW optimizer | ~5 GB |
| **Total** | **~31–37 GB ✓** |

**Optimizations:** 4-bit NF4 QLoRA, bf16 compute dtype, gradient checkpointing, 8-bit AdamW (`bitsandbytes`), `torch.compile`, Google Drive checkpoint mount every 500 steps

---

## Error Handling & Safeguards

- **Geodesic validity:** Assert $\xi \in \mathfrak{se}(3)$ and exp-map output $\in SE(3)$ after each ODE step; reject NaN/Inf trajectory batches
- **Gradient clipping:** `max_norm=1.0`; extra sensitivity on RFM head due to Lie algebra gradients
- **Loss monitoring:** Log each $\lambda_i \mathcal{L}_i$ separately; alert if any term exceeds 80% of total loss
- **Checkpoint resilience:** Lightning `ModelCheckpoint` every 500 steps + best-by-$\mathcal{L}_{RFM}$; auto-resume from Drive on Colab disconnect
- **Velocity clamping:** Generated SE(3) trajectories clipped to configurable max end-effector velocity
- **Workspace bounds:** Hard assert all generated poses lie within defined surgical workspace; violations logged and rejected
- **Danger zone override:** WARNING token from Text Head → RFM output zeroed, RETRACT gesture forced; no trajectory executed during active alert

---

## Test Plan

| Test | Scope | Checks |
|---|---|---|
| SE(3) log/exp roundtrip | Unit | `exp(log(T)) ≈ T` for 1000 random poses, error < 1e-6 |
| RFM ODE validity | Unit | 4-step geodesic ODE → valid SE(3) pose, no NaN, within workspace |
| Cross-modal alignment shape | Unit | All modality tokens project to identical hidden dim before Jamba |
| QLoRA injection | Unit | Jamba base params frozen; only LoRA adapters in optimizer |
| Gesture classifier | Unit | Output logits over full JIGSAWS vocabulary (10 classes) |
| Danger signal pathway | Integration | Injected force spike + low-confidence video → WARNING text + RETRACT gesture |
| Trajectory smoothness | Integration | Generated SE(3) path has bounded acceleration over 1–2s horizon |
| Long-context memory | Integration | Model recalls annotated vessel position after 500+ step history |
| Full inference pass | End-to-end | All 4 modalities in → trajectory + text + gesture out, latency < 100ms |

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| ATE (Absolute Trajectory Error) | SE(3) path accuracy vs. expert demonstrations (JIGSAWS) |
| Gesture recognition accuracy | Top-1 accuracy over JIGSAWS 10-class vocabulary |
| Geodesic smoothness (mean jerk) | Trajectory quality — lower is safer |
| Danger recall | % of injected danger signals correctly triggering WARNING + RETRACT |
| Text BLEU / BERTScore | Quality of generated surgical log text vs. reference notes |

---

## Stack

- **Framework:** PyTorch + PyTorch Lightning
- **Backbone:** Jamba (`ai21labs/Jamba-v0.1`), 4-bit NF4 QLoRA
- **Vision encoder:** CLIP ViT-L/14 (`openai/clip-vit-large-patch14`), frozen
- **Geometry:** Custom SE(3) log/exp map utilities (`pytorch3d` or manual implementation)
- **Policy head:** Riemannian Flow Matching DiT on SE(3)
- **QLoRA:** `peft` + `bitsandbytes` (`BitsAndBytesConfig` NF4, bf16 compute)
- **Optimizer:** 8-bit AdamW (`bitsandbytes`)
- **Hardware:** Google Colab A100 40GB
- **Datasets:** JIGSAWS, AutoLaparo, EndoVis 2017, synthetic SE(3) geodesics
- **Logging:** Weights & Biases or TensorBoard
