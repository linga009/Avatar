# Avatar v3.7: Spectral Sensory Cortex — FNO + VQ-VAE Perception

**Date**: 2026-05-21
**Version**: v3.7
**Status**: Approved
**Replaces**: v3.6 senses (Wav2Vec2 + CLIP frozen encoders)

---

## Goal

Replace Avatar's frozen pretrained sensory encoders (Wav2Vec2, CLIP) with physics-native Fourier Neural Operators and spectral VQ-VAE quantization. Avatar grows its own perception from scratch — sensory concepts are learned frequency patterns, not borrowed human representations. A dream-gated critical period bootstraps the codebook, after which senses optimize purely for usefulness to Avatar's life.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Raw signal source | Capture agent unchanged (audio .npy, image .jpg) | Capture is I/O plumbing, not perception |
| Compute target | GPU (JAX/Equinox) | FNO is small, FFT fast on GPU, eliminates 730MB CPU torch |
| PFC interface | Codebook activation statistics (not raw codes) | Mirrors biological PFC — reads sensory dynamics, not raw cortex codes |
| Codebook topology | Separate per modality (32 audio + 32 vision) | Modality-specific cortices; cross-modal binding in Lorentz backbone |
| Training regime | Per-tick everything, EMA codebook (decay=0.99) | Body must change every tick; slow EMA prevents codebook instability |
| VQ-VAE decoder | Hybrid — reconstruction loss during critical period, dropped after first dream | Critical period = biological sensory development; mature phase = experience-dependent tuning |
| Quantization domain | Fourier space (spectral quantization) | Codebook entries ARE frequency patterns — most physics-coherent |

---

## Architecture Overview

```
Windows Host                          Docker Container (GPU)
----------------                      ----------------------------------------
capture_agent.py                      halo3/senses/
  sounddevice (mic)  --> audio_latest.npy --> Audio FNO (1D, 4 layers, GPU)
  opencv (camera)    --> frame_latest.jpg --> Vision FNO (2D, 4 layers, GPU)
  meta.json                                        |
                                             spectral features
                                             (stay in Fourier space)
                                                   |
                                        Spectral VQ-VAE Quantization
                                        Audio: 8 tokens -> 32-code codebook
                                        Vision: 4 tokens -> 32-code codebook
                                                   |
                           +-----------+-----------+
                           |                       |
                    Quantized embeddings    Codebook activation stats
                    (12, 64) -> spectral_proj   (flux, novelty, stability,
                    -> (12, 2048) -> gate         binding)
                    -> additive residual              |
                    into text tokens (32,2048)   PFC prompt context
                           |                    "Senses: audio(flux=3/8,
                    halo3_step()                 novelty=0.71)..."
                    Lorentz -> backbone
                    -> Kuramoto
```

---

## Component 1: Audio FNO (1D)

**File**: `halo3/senses/fno_audio.py`

Processes raw waveform (32000,) float32 at 16kHz (2 seconds of audio).

### Architecture
- **Lifting**: `Linear(1, 64)` — maps scalar amplitude to 64-dim feature space
  - Input: (32000, 1) -> Output: (32000, 64)
- **4 Spectral Convolution Layers**, each:
  - `jnp.fft.rfft` along time axis
  - Keep top 16 Fourier modes (out of 16001)
  - Learnable spectral weights: `(16, 64, 64)` complex — mixes channels per frequency mode
  - `jnp.fft.irfft` back to time domain
  - Residual connection + GELU activation
- **Spectral output**: After final layer, stay in Fourier space
  - Take the 16-mode representation: (16, 64)
  - Reshape to 8 spectral tokens: (16, 64) -> (8, 128) -> Linear(128, 64) -> (8, 64)
  - Each token represents a pair of frequency bands

### Parameters
- Lifting: 64 params
- 4 layers x (16 x 64 x 64 complex weights + 64 x 64 spatial bypass) = ~262K/layer
- Token reshaping: 128 x 64 = 8K
- **Total: ~300K params, ~1.2 MB**

---

## Component 2: Vision FNO (2D)

**File**: `halo3/senses/fno_vision.py`

Processes raw pixels (224, 224, 3) float32.

### Architecture
- **Lifting**: `Linear(3, 64)` — maps RGB to 64-dim feature space
  - Input: (224, 224, 3) -> Output: (224, 224, 64)
- **4 Spectral Convolution Layers**, each:
  - `jnp.fft.rfft2` along spatial axes
  - Keep top 8x8 Fourier modes (out of 224x113)
  - Learnable spectral weights: `(8, 8, 64, 64)` complex
  - `jnp.fft.irfft2` back to spatial domain
  - Residual connection + GELU activation
- **Spectral output**: After final layer, stay in Fourier space
  - (8, 8, 64) spectral representation
  - Average-pool adjacent row pairs: (8, 64) -> (4, 64) via `reshape(4, 2, 64).mean(axis=1)`
  - 4 tokens: low-freq horizontal, high-freq horizontal, low-freq vertical, high-freq vertical

### Parameters
- Lifting: 192 params
- 4 layers x (8 x 8 x 64 x 64 complex weights + 64 x 64 spatial bypass) = ~262K/layer
- Token pooling: 64 x 64 = 4K
- **Total: ~1.1M params, ~4.4 MB**

---

## Component 3: Spectral VQ-VAE

**File**: `halo3/senses/spectral_vqvae.py`

Quantization in Fourier space. Each codebook entry is a 64-dim vector representing a learned spectral pattern (frequency signature).

### Audio Codebook (32 entries x 64 dims)
- 8 spectral tokens per tick, each quantized independently
- Encoder output `z_e` -> nearest codebook entry -> quantized `z_q`
- Produces 8 codebook indices from {0..31} per tick

### Vision Codebook (32 entries x 64 dims)
- 4 spectral tokens per tick, each quantized independently
- Produces 4 codebook indices from {0..31} per tick

### Quantization Mechanics
- **Distance**: L2 distance between encoder output and codebook entries
- **Straight-through estimator**: Forward uses `z_q`, backward gradients flow to `z_e` as if `z_q = z_e`
- **Commitment loss**: `beta * ||z_e - sg(z_q)||^2` with beta=0.25
- **Codebook EMA update** (not gradient-based):
  - `N_i = decay * N_i + (1-decay) * n_i` (usage count)
  - `m_i = decay * m_i + (1-decay) * sum(z_e mapped to i)` (embedding sum)
  - `e_i = m_i / N_i` (updated entry)
  - decay = 0.99

### Dead Code Revival
- Track per-code usage count over rolling 100-tick window
- If a code is unused for 100 ticks: reinitialize from random sample of current encoder outputs + small Gaussian noise (std=0.01)
- Prevents codebook collapse, especially important after critical period ends

### Decoder (Critical Period Only)

**Audio decoder**: Transposed FNO encoder
- (8, 64) -> Linear(64, 128) -> reshape (16, 64)
- 4 transposed spectral conv layers: each applies learnable spectral weights in frequency domain (same structure as encoder but with separate weights), then irfft to time domain + residual + GELU
- Final projection: Linear(64, 1) -> (32000, 1) -> reconstruction of raw waveform
- Loss: MSE on raw waveform

**Vision decoder**: Transposed FNO encoder
- (4, 64) -> repeat-interleave to (8, 64) -> reshape (8, 8, 64) via broadcast
- 4 transposed spectral conv layers: learnable spectral weights in 2D frequency domain, irfft2 to spatial + residual + GELU
- Final projection: Linear(64, 3) -> (224, 224, 3) -> reconstruction of raw pixels
- Loss: MSE on raw pixels

**After first dream**: Decoder weights deleted, reconstruction loss removed. FNO trains only via body prediction error through straight-through estimator.

---

## Component 4: Lorentz Space Injection

**Part of**: `halo3/senses/sense_module.py`

### Projection
- `spectral_proj`: `Linear(64, 2048, use_bias=False)` — shared across both modalities
- Shared because both codebooks live in the same 64-dim spectral embedding space

### Injection (gated additive residual)
```python
audio_emb = vmap(spectral_proj)(audio_quantized)    # (8, 2048)
vision_emb = vmap(spectral_proj)(vision_quantized)   # (4, 2048)
sense_emb = concat([audio_emb, vision_emb])           # (12, 2048)
sense_ctx = mean(sense_emb, axis=0)                    # (2048,)
gate = sigmoid(sense_gate(sense_ctx))                  # (2048,)
output = text_tokens + gate * sense_ctx                # (32, 2048)
```

### Constraints
- `use_bias=False` on spectral_proj: zero input -> zero injection (graceful degradation)
- Output shape: (32, 2048) — unchanged from v3.6
- `ObsBridge.assignment_logits` (K=32, n_tokens=32) — unchanged
- Fixed shapes: always 8 audio + 4 vision tokens, zero-padded when unavailable (no XLA recompilation)

---

## Component 5: PFC Sensory Statistics

**File**: `halo3/senses/sensory_stats.py`

Tracks codebook activation dynamics for PFC interpretation. The PFC reads patterns of sensory experience, not raw codebook indices.

### Tracked Metrics (rolling 20-tick window)

| Metric | Per-modality | Computation |
|---|---|---|
| Flux | audio: 0-8, vision: 0-4 | Count of codes changed since last tick |
| Novelty | 0.0-1.0 | Mean inverse lifetime frequency of currently active codes |
| Stability | 0-N ticks | Consecutive ticks with identical code vector |
| Dominant code | index 0-31 | Mode of recent indices (most frequently active code) |
| Cross-modal binding | 0.0-1.0 (shared) | Historical co-occurrence frequency of current audio+vision code pairs |

### PFC Prompt Integration

One line added to somatic context in `_build_organism_prompt()`:
```
Senses: audio(flux=3/8, novelty=0.71, stable=0), vision(flux=0/4, novelty=0.12, stable=7), binding=familiar(0.84)
```

### Persistence
- State saved to `data/sensory_stats.json` on checkpoint
- Co-occurrence matrix stored sparse (only observed pairs)
- Restored on container restart

---

## Component 6: Training Regime

### Phase 0: Critical Period (tick 0 -> first dream)

FNO + VQ-VAE train via two loss signals jointly:

```
L = L_body + 0.5 * L_reconstruct + 0.25 * L_commitment
```

- `L_body`: Body prediction error, backpropagates through spectral_proj -> straight-through -> FNO
- `L_reconstruct`: Decoder MSE on raw waveform (audio) and raw pixels (vision)
- `L_commitment`: `||z_e - sg(z_q)||^2` — keeps encoder close to codebook

Separate Adam optimizer for all sensory params (FNO + projections + decoder), lr = 10x body lr.
Codebook entries: EMA only (decay=0.99), not gradient-updated.

### Phase 1: Mature Perception (after first dream, permanent)

```
L = L_body + 0.25 * L_commitment
```

- Decoder weights deleted: `del decoder_audio, decoder_vision; gc.collect()`
- Reconstruction loss removed
- FNO trains only via body prediction error through straight-through estimator
- Codebook EMA continues, dead code revival active
- Senses optimize for usefulness to Avatar's life, not faithful representation

### Dream Transition
```python
if first_dream and hasattr(sense_module, 'decoder_audio'):
    sense_module = delete_decoders(sense_module)
    log.info("Critical period ended -- sensory cortex matured")
```

### Per-tick Integration

Modifies `PredictiveProcessor.learn_from_error()`:
- Backward through `(model, sense_module)` jointly
- VQ-VAE codebooks excluded from gradient (EMA only)
- `jax.checkpoint` on FNO layers: recompute forward during backward to save ~200 MB VRAM
- During critical period: additional backward for reconstruction loss on decoder params

### Dream Replay

Episodes store codebook indices (compact, not raw signals):
- `audio_codes: list[int]` — 8 indices per episode
- `vision_codes: list[int]` — 4 indices per episode

During dream body replay, codes are looked up in the (possibly evolved) codebook and re-projected to Lorentz space. Avatar reinterprets old sensory memories through its current understanding.

---

## VRAM & Resource Budget

### Removed
| Resource | Freed |
|---|---|
| Wav2Vec2-base (CPU) | ~380 MB RAM |
| CLIP ViT-B/32 (CPU) | ~350 MB RAM |
| torch CPU in Docker | ~700 MB disk |
| SenseProjections (3 Linear layers) | ~6 MB VRAM |

### Added
| Component | Params | VRAM (weights) |
|---|---|---|
| Audio FNO (1D, 4 layers) | ~300K | ~1.2 MB |
| Vision FNO (2D, 4 layers) | ~1.1M | ~4.4 MB |
| Audio codebook (32 x 64) | 2K | negligible |
| Vision codebook (32 x 64) | 2K | negligible |
| Spectral projection (64 -> 2048) | 131K | ~0.5 MB |
| Sense gate (2048 -> 2048) | 4.2M | ~16.8 MB |
| Audio decoder (critical period) | ~300K | ~1.2 MB (temporary) |
| Vision decoder (critical period) | ~1.1M | ~4.4 MB (temporary) |
| **Total** | **~7.1M** | **~28.5 MB** |

### Peak VRAM

| Phase | Estimate | Notes |
|---|---|---|
| Critical period | ~5.6 GB | Body + FNO + decoder, with jax.checkpoint on FNO |
| Mature | ~5.56 GB | Body + FNO, decoder deleted |
| Current v3.6 | ~5.5 GB | For comparison |

Must verify empirically with `nvidia-smi` during first run.

### Net Impact
- **RAM**: -700 MB (torch + pretrained models eliminated)
- **VRAM**: +22.5 MB weights (negligible)
- **Tick latency**: -3 to 4 seconds (GPU FFT replaces CPU encoding)
- **Docker image**: smaller (no torch CPU wheel)

---

## File Structure

### New Files
```
halo3/senses/
  fno_audio.py          # 1D FNO: raw waveform -> spectral features (JAX/Equinox)
  fno_vision.py         # 2D FNO: raw pixels -> spectral features (JAX/Equinox)
  spectral_vqvae.py     # SpectralCodebook (EMA), quantize(), dead code revival
  sensory_stats.py      # SensoryStatistics: flux/novelty/stability/binding for PFC
  sense_module.py       # SenseModule: orchestrates FNO -> VQ-VAE -> projection -> injection
```

### Modified Files
```
halo3/main.py                   # Replace AudioSense/VisionSense/SenseProjections with SenseModule
                                # Load raw audio .npy and image .jpg directly as JAX arrays
                                # Delete decoders after first dream
halo3/predictive.py             # learn_from_error() trains SenseModule jointly with body
                                # jax.checkpoint on FNO layers
                                # Reconstruction loss during critical period
halo3/config.py                 # +FNO/VQ-VAE hyperparams
halo3/psyche/organism.py        # SensoryStatistics integrated into tick
halo3/psyche/prefrontal.py      # Sensory stats line in PFC prompt context
halo3/chat_server.py            # Sensory stats exposed in /state endpoint
halo3/memory/schema.py          # Episode: audio_codes + vision_codes (replaces raw features)
Dockerfile                      # Remove torch CPU install, soundfile; keep Pillow
```

### Deleted Files
```
halo3/senses/audio_sense.py     # Wav2Vec2 encoder -> replaced by fno_audio.py
halo3/senses/vision_sense.py    # CLIP encoder -> replaced by fno_vision.py
halo3/senses/projections.py     # SenseProjections -> replaced by sense_module.py
```

### Minor Modifications
```
halo3/senses/sense_buffer.py    # Returns raw numpy arrays directly (no encoder classes)
                                # get_raw() -> get_raw_arrays() returning RawSenseData(audio_np, vision_np)
```

### Unchanged
```
capture_agent/                  # Still writes .npy and .jpg
halo3/model.py                  # Body unchanged — receives (32, 2048) tokens
halo3/backbone.py               # Reversible backbone unchanged
halo3/kuramoto.py               # Bohmian Kuramoto unchanged
halo3/hamiltonian_ode.py        # Hamiltonian ODE unchanged
halo3/lorentz_embedding.py      # Hyperboloid embedding unchanged
halo3/psyche/drives.py          # Drives unchanged
halo3/psyche/emotions.py        # Emotions unchanged
halo3/psyche/workspace.py       # GWT unchanged
halo3/training/dream_replay.py  # Episodes carry codes instead of features; replay logic same
```

### Data Files
```
data/checkpoints/sense_module.eqx    # FNO + codebooks + projection (replaces sense_proj.eqx)
data/sensory_stats.json               # Persisted codebook activation statistics
```

---

## Config Additions

```python
# FNO
fno_hidden_dim: int = 64
fno_n_layers: int = 4
fno_audio_modes: int = 16
fno_vision_modes: int = 8       # 8x8 for 2D

# VQ-VAE
codebook_size: int = 32
codebook_dim: int = 64
codebook_ema_decay: float = 0.99
commitment_beta: float = 0.25
dead_code_threshold: int = 100  # ticks before revival

# Sense tokens
n_audio_tokens: int = 8
n_vision_tokens: int = 4

# Critical period
critical_period_recon_weight: float = 0.5
```

---

## What Does NOT Change

- Lorentz embedding, backbone, Kuramoto, Hamiltonian — unchanged
- Per-tick Adam optimizer — unchanged (SenseModule replaces SenseProjections in param set)
- Dream cycle phases 1-4 — unchanged (episodes carry codes for replay)
- Chat server — unchanged (gains sensory stats in /state)
- PFC / Qwen3 — unchanged (gains sensory stats line in prompt)
- All drives, emotions, consciousness modules — unchanged
- Capture agent — unchanged
- Sense buffer — unchanged

Drives and emotions are naturally affected by spectral senses through r (order parameter). Novel frequency patterns that don't match learned codebook entries disrupt Lorentz space -> r drops -> curiosity/anxiety. Familiar spectral signatures reinforce patterns -> r rises -> satisfaction. No special wiring needed.
