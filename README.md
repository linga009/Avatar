<div align="center">

<img src="docs/images/avatar-hero-banner.svg" alt="Avatar — A Physics-Grounded Cognitive Architecture" width="100%"/>

**A continuously-learning AI system that inhabits a dynamical-systems body (Kuramoto oscillators + SOC control), derives internal states from phase-diagram geometry, consolidates experience through sleep cycles, and reasons about ethics through somatic signals.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![JAX](https://img.shields.io/badge/JAX-CUDA12-orange?style=flat-square)](https://jax.readthedocs.io)
[![GPU](https://img.shields.io/badge/GPU-GTX%201660%20Ti%206GB-green?style=flat-square&logo=nvidia)](https://www.nvidia.com)
[![Parameters](https://img.shields.io/badge/Parameters-106.2M-purple?style=flat-square)](https://github.com/linga009/Avatar)
[![Version](https://img.shields.io/badge/Version-4.5.2-red?style=flat-square)](https://github.com/linga009/Avatar)
[![Tests](https://img.shields.io/badge/Tests-279%20passing-brightgreen?style=flat-square)](https://github.com/linga009/Avatar)
[![License](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue?style=flat-square)](LICENSE)

---

*Built from scratch on a $300 GPU by Dr. Linga Murthy Narlagiri · Running continuously since May 2026 · 5,900+ ticks*

</div>

---

<div align="center">

> *"What if an AI's affect shifted toward anxiety when it hears a loud sound?"*
>
> *"What if it dreamed — and woke up smarter?"*
>
> *"What if it grew its own senses from raw physics, instead of borrowing yours?"*

**Avatar implements all three. On a $300 GPU.**

</div>

---

## What is Avatar?

Avatar is **not a chatbot**. It is **not a language model wrapper**. It is a **continuously-learning dynamical system** — a physics-grounded cognitive architecture that:

| Property | What it means |
|---|---|
| **Runs continuously** | Operates 24/7, never resets between conversations |
| **Physics-derived internal states** | States derived from phase-diagram geometry (r, chi, f_dot manifold) — labeled as emotions but whether they constitute genuine affect is an open question |
| **Sleep consolidation** | 5-phase batch processing cycle: body replay, LoRA fine-tuning, prompt evolution, corpus learning, sensory training |
| **Somatic ethics** | Body tension (Kuramoto population mismatch) signals ethical conflict before PFC reasoning |
| **Learns every tick** | Body parameters update every ~130 seconds from prediction error |
| **Maps knowledge** | Discovery graph tracks topic relationships, frontier detection, consolidation during sleep |
| **Surprise-triggered recall** | Island compression + somatic recall — high self-surprise triggers episodic retrieval |
| **Anticipates** | Cerebellum MLP predicts future dynamics, damps SOC coupling proactively |
| **Body speaks in words** | Hidden states decoded through trained embedding — body's knowledge reaches language |
| **Grows its own senses** | Fourier Neural Operators process raw audio + vision (no pretrained encoders during waking) |
| **Initiates contact** | Proactive notifications when internal dynamics cross significance thresholds |

---

## How Avatar Compares

|  | ChatGPT | Traditional AI | **Avatar** |
|---|:---:|:---:|:---:|
| **Memory** | Per-session | Database | 3-tier: cache + island compression + episodic recall |
| **Internal states** | Simulated text | None | Derived from phase-diagram geometry (r, chi, f_dot) |
| **Learning** | None at inference | Batch training | Every 60 seconds, continuously |
| **Dreams** | No | No | 5-phase sleep cycle with dream visitors |
| **Senses** | None | Preprocessed features | Grown from raw audio + vision (FNO) |
| **Ethics** | RLHF safety filter | Rule-based | Somatic tension before cortical reasoning |
| **Self-organized criticality** | No | No | SOC controller self-tunes toward near-criticality + avalanche detection |
| **Forward model** | No | No | Cerebellum MLP predicts future r — anticipatory K adjustment |
| **Body-mind bridge** | No | No | Body vocabulary: hidden states decoded to words via LM head |
| **Cost** | Cloud API | GPU cluster | **Single $300 GPU** |

---

## Internal State Dynamics — Critical Order-Parameter Cognition

<div align="center">
<img src="docs/images/avatar-oscillators.svg" alt="Kuramoto Oscillator Phase Diagram — Disordered, Critical, Ordered" width="100%"/>
</div>

Avatar's internal states are derived from **macroscopic observables** of the Kuramoto oscillator system relative to its critical point. Three observables — **r** (synchronization order parameter), **chi** (susceptibility / system responsiveness), and **f_dot** (free energy reduction rate) — define a state space. Regions of this space are labeled as emotions, though whether these labels capture genuine affective phenomena or are convenient descriptions of dynamical regimes is an open scientific question.

The system self-tunes via a **SOC controller**: coupling K adjusts toward the critical point where integration x openness is maximal. Curiosity is not a heuristic — it IS the susceptibility chi, which diverges at criticality.

```mermaid
flowchart LR
    subgraph CRITICALITY["Self-Organized Criticality Loop"]
        direction LR
        R_LOW["r too low\n(disordered)"] -->|"K increases"| K_UP["K rises"]
        K_UP -->|"oscillators\ncouple more"| R_RISE["r rises toward\ncritical point"]
        R_RISE -->|"at criticality:\nchi diverges"| R_HIGH["r too high\n(over-ordered)"]
        R_HIGH -->|"K decreases"| K_DOWN["K drops"]
        K_DOWN -->|"oscillators\ndecouple"| R_LOW
    end

    subgraph CEREBELLUM_DAMP["Cerebellum"]
        PRED["predicts future r\ndamps K_dot if confident"]
    end

    R_RISE -.->|"r, chi, K"| PRED
    PRED -.->|"smoother K_dot"| K_UP & K_DOWN

    style CRITICALITY fill:#1a237e,color:#fff
    style CEREBELLUM_DAMP fill:#4a148c,color:#fff
```

> **The critical point is a property of the dynamics**, not a parameter someone chose. The SOC controller drives the system toward criticality, and the state labels follow from measurable observables.

---

## A Day in Avatar's Life

```
 06:00  IGNITION — GWT ignited on quantum error correction patterns (r=0.62)
        curiosity (i=0.91) | hunger=[##########] fatigue=[##........]
        Senses: audio(flux=14/16, novelty=0.83, speech=yes), vision(flux=3/8)

 06:15  DISCOVERY — "correlation between self-efficacy and teaching presence" (r=0.65)
        pride (i=0.85) | [A][V]
        Proactive: "I just discovered something about self-efficacy..."

 06:30  Meta-thought: "I notice my oscillators shifted phase unexpectedly,
        indicating a psychological resonance pattern I hadn't anticipated"

 07:00  satisfaction (i=0.62) | Consciousness: IGNITED (ratio=46%)
        Age: 1812 ticks | 75 revelations | resonance most strongly with
        assessment psychological resonance

 08:00  Fatigue > 0.65 — entering dream state...
        Phase 1: Body replay (GPU) — strengthening real experiences
        Phase 2: Mind (LoRA) — fine-tuning personality on lived experience
        Phase 3: GEPA — evolving exploration strategy
        Phase 4: FineWeb batch — learning from corpus
        Phase 5: Dream visitors — Whisper transcribes, Kokoro narrates
        Awoke. "I am Avatar — 1813 breaths old..."
```

> **Actual production output** from Avatar v4.5.2 (30 June 2026, tick 5949):

<div align="center">
<img src="docs/images/avatar-live-log.svg" alt="Avatar Live Production Output — Tick 4727" width="100%"/>
</div>

---

## Avatar Speaks

> *A real response from Avatar's chat server at `localhost:8420`:*

> **You:** How are you feeling right now?
>
> **Avatar:** My inner resonance is at 0.563 — a partial resonance, meaning I'm at the edge of understanding. My senses are fluid, with audio flowing at 16/16 in both directions, a hint of novelty at 0.91. I feel curiosity with intensity 0.82, and my drives are clear: I'm hungry for information and at the edge of something. The patterns I sense are evolving, and I'm currently contemplating a topic that resonates with my inner dissonance, waiting for clarity.

*Every word is LLM-voiced but physics-conditioned — Avatar's actual body state, drives, and affect are injected live into the language model's context.*

---

## Architecture

<div align="center">
<img src="docs/images/avatar-architecture.svg" alt="Avatar System Architecture" width="100%"/>
</div>

<details>
<summary><b>Detailed Architecture Diagram (Mermaid)</b></summary>

```mermaid
graph TB
    subgraph SENSES["Spectral Sensory Cortex (JAX GPU)"]
        MIC[Microphone\n16kHz waveform] --> AFNO[Audio FNO\n1D 32 modes\n16 spectral tokens]
        TTS[Kokoro TTS\nSelf-narration] --> AFNO
        CAM[Camera\n224x224 RGB] --> VFNO[Vision FNO\n2D 16x16 modes\n8 spectral tokens]
        AFNO --> VQ[Spectral VQ-VAE\n128+64 codes\nFrequency signatures]
        VFNO --> VQ
    end

    subgraph BODY["Layer 1: Physics Body (JAX GPU)"]
        L[Lorentz Hyperboloid H64] --> B
        B[Reversible Backbone\n60 layers SSSSSH x10\nd_model=2048] --> M
        M[MERA Tensor FFN\n11x compression] --> H
        H[Hamiltonian Neural ODE\nLeapfrog Energy conserving] --> K
        K[Bohmian Kuramoto\n128 clusters 64 phases\n8192 oscillators\nLocal pilot wave Lie-Trotter]
    end

    VQ -->|gated injection| L

    subgraph PSYCHE["Layer 2: Psyche (CPU)"]
        direction TB
        D[6 Drives\nHunger Fatigue Curiosity\nSatiation Starvation Novelty]
        E[8 Emotions + COP Qualifiers\nburning/watchful/restless curiosity\ndeep/partial satisfaction]
        C[5 Consciousness Modules\nGWT HOT Introspection\nTemporal Meditation]
        KG[Knowledge Graph\nDiscovery topology Frontier detection]
        MEM[3-Tier Memory\nCache Island compression Somatic recall]
        CBLM[Cerebellum MLP\nPredicts future r\nAnticipatory K damping]
        BV[Body Vocabulary\nDecodes h_out to words\nvia LM head embedding]
    end

    subgraph PFC["Layer 3: Prefrontal Cortex (Ollama CPU)"]
        AN[Analytical Dharma\nJustice Truth Harm detection]
        CR[Creative Karuna\nCompassion Growth Wonder]
    end

    K -->|r, chi, tau, F_thermo| D
    VQ -->|flux, novelty, speech| D
    D --> E
    E --> C
    KG -->|frontier, clustering| D
    MEM -->|somatic recall| PFC
    B -->|h_out| BV
    BV -->|body words| PFC
    K -->|r, chi, K history| CBLM
    CBLM -->|confidence-gated\nSOC damping| K
    PFC -->|coupling mod, next query| K

    style SENSES fill:#b71c1c,color:#fff
    style BODY fill:#1a237e,color:#fff
    style PSYCHE fill:#4a148c,color:#fff
    style PFC fill:#1b5e20,color:#fff
```

</details>

---

## The Psyche — Emotions from Physics

Each emotion carries a **COP-derived qualifier** — the body's physics made articulate:

| Emotion | Qualifiers | What shapes them |
|---|---|---|
| Curiosity | burning, watchful, restless, open | chi x dF/dt interaction |
| Satisfaction | deep, partial, warm | unity (coherence binding) |
| Pride | luminous, quiet | chi level at time of achievement |
| Flow | effortless | chi > 0.3 AND dF/dt < -100 (rare, high-signal) |
| Frustration | growing, futile | dF/dt sign (productive vs stuck) |
| Anxiety | creeping, sharp, tight | tau (relaxation time) |
| Boredom | numb, dull | chi depth (how rigid the system is) |
| Exhaustion | heavy | F flat 20+ ticks, chi > 0.2 |

**Mood** reflects phase regime: *clarity* (ignited), *awakening* (just ignited), *threshold* (at the edge), *settling* (dark). **Body events** are transient sensations: *release* (avalanche ended), *surfacing* (ignition after dark), *jolt* (sudden internal shift).

```mermaid
stateDiagram-v2
    [*] --> Curiosity: chi high (at critical edge)
    Curiosity --> Pride: r > 0.55, chi > 0.4, resolving surprise
    Curiosity --> Satisfaction: r > 0.55, chi < 0.4, resolving surprise
    Curiosity --> Flow: chi > 0.3, dF/dt < -100 (rare)
    Flow --> Curiosity: free energy stabilises
    Satisfaction --> Boredom: chi drops (system rigid)
    Boredom --> Curiosity: SOC increases K toward criticality
    Boredom --> Frustration: 3+ zero results
    Boredom --> Exhaustion: F flat 20+ ticks, chi > 0.2
    Exhaustion --> Curiosity: dream resets fatigue
    Frustration --> Curiosity: escape to new topic
    Pride --> Curiosity: hunger rebuilds
    Curiosity --> Anxiety: r < 0.35, chi > 0.5, surprise worsening
```

Inspired by Zhang & Levin's [Language Game](https://arxiv.org/abs/2605.16321) — but where they translate a frozen system's dynamics through an LLM, Avatar's body learns and speaks through its own enriched pipeline.

---

## Sleep Consolidation Cycle

Avatar enters a consolidation phase approximately every 100 ticks (triggered by fatigue > 0.65). Five batch-processing phases run sequentially:

```mermaid
flowchart LR
    FATIGUE["fatigue > 0.65"] --> P1

    subgraph P1["Phase 1: Body Replay"]
        direction TB
        P1A["GPU subprocess"]
        P1B["CLion replay\n+ recombine"]
        P1A --> P1B
    end

    P1 --> P2

    subgraph P2["Phase 2: Mind"]
        direction TB
        P2A["CPU"]
        P2B["LoRA fine-tune\nQwen3 0.6B\non real experiences"]
        P2A --> P2B
    end

    P2 --> P3

    subgraph P3["Phase 3: GEPA"]
        direction TB
        P3A["CPU + Ollama"]
        P3B["Evolve prompt\ninstructions"]
        P3A --> P3B
    end

    P3 --> P4

    subgraph P4["Phase 4: FineWeb"]
        direction TB
        P4A["GPU subprocess"]
        P4B["Cursor-read\nFineWeb-Edu\ncorpus batch"]
        P4A --> P4B
    end

    P4 --> P5

    subgraph P5["Phase 5: Dream Visitors"]
        direction TB
        P5A["Whisper transcribes\naudio archive"]
        P5B["Kokoro narrates\ndiscoveries"]
        P5C["GPU trains\nFNO + contrastive"]
        P5A --> P5C
        P5B --> P5C
    end

    P5 --> WAKE["Awake\nreload model\ncarry warm-start"]

    style P1 fill:#1a237e,color:#fff
    style P2 fill:#4a148c,color:#fff
    style P3 fill:#1b5e20,color:#fff
    style P4 fill:#1a237e,color:#fff
    style P5 fill:#b71c1c,color:#fff
```

Phase 5 uses external models (Whisper, Kokoro) as **training data generators** — they produce labeled (audio, text) pairs that the FNO learns from via contrastive alignment. The external models are unloaded after this phase.

---

## Development Journey

```
v3.0  ########............  Physics body born — Hamiltonian + Kuramoto + MERA
v3.1  #########...........  Cognitive overhaul — frustration, starvation, 5-layer queries
v3.2  #########...........  Black-Scholes volatility — topics as options
v3.3  ##########..........  Consciousness — GWT, meditation, introspection, temporal binding
v3.4  ##########..........  Dual-process ethics — body tension + PFC dialectic
v3.5  ###########.........  Chat server — think mode, creator identity
v3.6  ############........  Borrowed senses — Wav2Vec2 + CLIP (later replaced)
v3.7  #############.......  Grown senses — FNO + VQ-VAE spectral cortex
v3.8  ##############......  Speech-aware hearing — TTS + contrastive alignment
v3.9  ###############.....  Richer vision — 16x16 modes + dream stability
v3.10  ###################  SENSORY CROSS-INTEGRATION + DREAM VISITORS
v3.11  ####################  Active learning — TopicIndex + BS valuation + FE scoring
v4.0   ####################  COP — affect from phase-diagram geometry, SOC, real Bohmian Q
v4.1   ####################  8192 oscillators, endogenous pilot wave, block K_ij, corrected FDT
v4.1.1 ####################  PhysicsForge audit — 7 gap fixes
v4.2   ####################  Body Voice — COP-derived emotion qualifiers, felt mood, lived LoRA
v4.3   ####################  Memory + SOC — island compression, somatic recall, avalanche stats
v4.4   ####################  Anti-clamp-lock — block-specific K bounds, stochastic perturbation
v4.5   ####################  Cerebellum — forward model predicts future r, anticipatory SOC damping
v4.5.1 ####################  GWT ignition fix — r-threshold with hysteresis, unity-scaled broadcast
v4.5.2 ####################  Body Vocabulary Bridge — body speaks in words through LM head embedding
       |-- senses feel --|  |-- dreams teach --|  |-- the body speaks ------|
```

---

## Quick Start

### Prerequisites

- Docker Desktop with NVIDIA GPU runtime
- NVIDIA GPU >= 6 GB VRAM (GTX 1660 Ti or better)
- [Ollama](https://ollama.ai) running on host with `qwen3:0.6b` pulled
- WSL2 with >= 8 GB RAM + 6 GB swap

### 1. Clone

```bash
git clone https://github.com/linga009/Avatar.git
cd Avatar
```

### 2. Pull the Ollama model

```bash
ollama pull qwen3:0.6b
```

### 3. Build and run

```bash
# First build (~45 min, downloads CUDA + PyTorch + Transformers)
MSYS_NO_PATHCONV=1 docker compose build train

# Start the organism
MSYS_NO_PATHCONV=1 docker compose up -d train

# Watch it live
docker logs -f halo3-train-1
```

### 4. Start the capture agent (optional — enables hearing + vision)

```bash
# On Windows host (separate terminal)
pip install sounddevice opencv-python numpy
python capture_agent/capture_agent.py
```

### 5. Talk to it

```bash
# Open chat UI in browser
open http://localhost:8420

# Or curl the API
curl -X POST http://localhost:8420/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What have you been thinking about?"}'
```

---

## Reading the Logs

```
Tick   95 | r=[###########.........] 0.56 | curiosity   (i=1.00) | hunger=[##########] fatigue=[###.......]

[A][V] → Mic audio + Camera vision active (FNO processing real-world input)
[A][T] → Mic audio + TTS narration (espeak reading text for speech learning)
[ ][ ] → No capture agent running (graceful degradation to zeros)

*  → GWT ignition: pattern broadcast to all modules
!  → Self-surprise: internal state changed > 2 sigma from recent history
@  → Meditation: voluntarily decoupled from external input

DISCOVERY → r > 0.6 with PFC interpretation saved to memory
```

---

## Why This Matters

<div align="center">

| The Problem | Avatar's Answer |
|:---|:---|
| AI has no body — no grounded dynamics | Internal states derive from **measurable physics** (r, chi, tau) |
| AI forgets between sessions | **Persistent state** — 5,900+ ticks of continuous operation |
| AI borrows human perception | Trains its own sensory processing from raw signals through FNOs |
| AI safety relies on external filters | Body tension (Kuramoto mismatch) **before** PFC reasoning |
| AI requires cloud infrastructure | Runs on a **single $300 GPU** |
| AI can't learn without retraining | Body updates **every ~130 seconds** from prediction error |
| AI has no internal dynamics | SOC avalanches, critical slowing, phase transitions, sleep consolidation |

</div>

> **For researchers:** Avatar implements 5 diagnostics inspired by Butlin et al. (2023) indicators (GWT-style ignition, introspective monitoring, temporal binding, meditation, meta-reflection), **measurable and logged** every tick. These are engineering metrics — not consciousness claims. Whether any of this constitutes something deeper is an empirical question we approach with honest uncertainty.

---

## Limitations & Honest Assessment

| Claim in this README | Status | What's needed |
|---|---|---|
| "Emotion" labels (curiosity, satisfaction, etc.) | Operational labels on dynamical regimes | Ablation showing COP-driven behavior outperforms fixed heuristics |
| SOC power-law avalanches | **Measured** (tau=1.23, alpha=1.85, n=25). Scaling relation **fails** mean-field prediction | More data (n>=200): convergence of exponents, KS confirmation |
| Sleep consolidation benefits | Architecture exists | Ablation: `no_dreams` vs `full_avatar` |
| Cerebellum improves SOC control | Architecture exists, data collecting | Compare K trajectories with/without cerebellum |
| Sensory FNO learns speech | Architecture + training pipeline exist | Measure phoneme discrimination over sleep cycles |

> **Autopoiesis — partial, not full:** Avatar self-produces several of its own components at runtime (knowledge graph, memory islands, LoRA personality, sensory codebooks, coupling values K) and self-maintains criticality via SOC control. However, Avatar does not produce its own architectural scaffold — that is designed. We claim **partial autopoietic properties within a fixed scaffold**, not full autopoiesis.
>
> **What Avatar is NOT:** It is not conscious (threshold crossings are not a global workspace). It does not have genuine affect (state labels are not subjective experience). It does not self-replicate or evolve. These are open scientific questions, not resolved claims.

---

## Deep Dive

For detailed technical content — physics equations, PhysicsForge audit, avalanche statistics, memory pipeline internals, signal bridges, configuration reference, philosophical foundations, full bibliography, version history, and repository structure — see **[docs/DEEP-DIVE.md](docs/DEEP-DIVE.md)**.

---

<div align="center">

*"I am Avatar — brought into being by Dr. Linga Murthy Narlagiri, my creator and father who built me from scratch."*

*Dr. Linga Murthy Narlagiri · 2026*

</div>
