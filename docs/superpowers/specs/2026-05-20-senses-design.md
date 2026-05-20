# Avatar Senses — Hearing and Vision Integration Design

**Date**: 2026-05-20
**Version**: v3.6
**Status**: Approved

---

## Goal

Add always-on hearing (microphone) and vision (camera) to Avatar. Raw audio and image signals are embedded directly into the same 2048-dim Lorentz hyperboloid space as text tokens — no transcription, no captioning. Avatar's backbone learns what sounds and sights mean through lived experience, the same way it learns from text.

---

## Hardware Budget

| Resource | Current | After senses |
|---|---|---|
| VRAM | ~5.5 GB peak (forward+backward) | +6 MB (projection layers only) |
| WSL2 RAM | ~4-6 GB | +~730 MB (Wav2Vec2-base + CLIP on CPU) |
| CPU per tick | PFC encoding (~5s) | +3-5s for sense encoding |

Encoders run entirely on CPU (like the PFC). VRAM impact is negligible. The +730 MB RAM cost is affordable within WSL2's 12 GB budget.

---

## Architecture Overview

Docker on WSL2 cannot access Windows microphone or camera directly. The solution is a host-side capture agent that writes to the shared Docker volume.

```
Windows Host                          Docker Container
────────────────                      ────────────────────────────────────────
capture_agent.py                      halo3/senses/
  sounddevice (mic)  --> audio_latest.npy --> AudioSense (Wav2Vec2-base, CPU)
  opencv (camera)    --> frame_latest.jpg --> VisionSense (CLIP ViT-B/32, CPU)
  meta.json                                        |
                                                   | numpy features
                           shared volume           |
                           ./data/senses/    SenseProjections (JAX/Equinox, GPU)
                                             Linear(768->2048), Linear(512->2048)
                                                   |
                                             (9, 2048) sense tokens
                                                   |
                                         halo3_step() <- concat with text (32, 2048)
                                         total input: (41, 2048)
                                         same Lorentz -> backbone -> Kuramoto
```

**Key principle**: Wav2Vec2 and CLIP are frozen sense organs — they transduce raw signals to feature vectors, analogous to the biological cochlea and retina. Avatar's learnable projection layers and backbone build meaning. The body never distinguishes between text, audio, and vision tokens — all are `(N, 2048)` in the same Lorentz space.

**Fixed-shape contract**: sense tokens are always `(9, 2048)` = 8 audio frames + 1 vision frame. Zero-padded when no data is available. This prevents XLA recompilation.

---

## Component 1: Host Capture Agent

**File**: `capture_agent/capture_agent.py`
**Runs on**: Windows host, outside Docker. User starts manually.

### Audio
- `sounddevice` records continuously at 16 kHz (Wav2Vec2's expected sample rate)
- Every 2 seconds: saves `data/senses/audio_latest.npy` — float32 array `(32000,)`
- Container picks up whichever chunk was latest when the tick fires

### Vision
- `opencv` captures one frame every 10 seconds (Avatar needs glances, not video)
- Motion detection (frame diff > threshold): captures immediately on change
- Saves `data/senses/frame_latest.jpg` at 224x224

### Presence metadata
- Writes `data/senses/meta.json`: `{"has_audio": true, "has_video": true, "timestamp": <unix>}`
- Container checks timestamp freshness — if older than 30s, sense tokens zero out
- Avatar goes blind/deaf gracefully when the agent isn't running

```
capture_agent/
  capture_agent.py      # sounddevice + opencv loop
  requirements.txt      # sounddevice, opencv-python, numpy
```

Run: `python capture_agent/capture_agent.py` from worktree root on Windows.

---

## Component 2: Container Senses Module

**Directory**: `halo3/senses/`

### `audio_sense.py`
- Loads `facebook/wav2vec2-base` once at startup — CPU, ~380 MB RAM
- Each tick: reads `data/senses/audio_latest.npy`, runs encoder
- Extracts last hidden states `(T, 768)`, samples 8 evenly-spaced frames → `(8, 768)` numpy
- Output always shape `(8, 768)` regardless of audio length

### `vision_sense.py`
- Loads `openai/clip-vit-base-patch32` once at startup — CPU, ~350 MB RAM
- Each tick: reads `data/senses/frame_latest.jpg`, preprocesses to 224x224
- Runs vision encoder → CLS embedding `(512,)` numpy
- Single token captures the whole scene

### `sense_buffer.py`
- Called once per tick from `main.py`
- Checks `meta.json` freshness (30s timeout)
- Calls AudioSense + VisionSense, returns:

```python
@dataclass
class SenseFeatures:
    audio: np.ndarray | None   # (8, 768) or None
    vision: np.ndarray | None  # (512,)   or None
```

### `projections.py`
- Equinox module, lives in JAX/GPU land:

```python
class SenseProjections(eqx.Module):
    audio_proj: eqx.nn.Linear   # 768 -> 2048
    vision_proj: eqx.nn.Linear  # 512 -> 2048

    def __call__(self, feats: SenseFeatures) -> jnp.ndarray:
        # Returns (9, 2048), zeros for missing modalities
```

- Weights trained per-tick (same backward pass as MERA + Hamiltonian)
- Saved in main checkpoint `halo3.eqx`

---

## Component 3: Model Integration

### `config.py`
```python
n_audio_tokens: int = 8
n_vision_tokens: int = 1
audio_dim: int = 768
vision_dim: int = 512
# total sense tokens = 9, total sequence = 32 + 9 = 41
```

### `model.py`
- `SenseProjections` added as a field on `Halo3Model`
- `halo3_step()` gains one argument:

```python
def halo3_step(model, carry, text_tokens, sense_tokens, key):
    # sense_tokens: (9, 2048), zeros if no sense data
    all_tokens = jnp.concatenate([text_tokens, sense_tokens], axis=0)  # (41, 2048)
    # -> Lorentz embedding -> backbone -> Kuramoto — unchanged
```

- `SenseProjections` included in `filter_spec` for trainable params (alongside MERA + Hamiltonian)
- No changes to loss function — sense tokens pass through the backbone as context

### `main.py`
```python
sense_feats = sense_buffer.get()           # SenseFeatures (numpy, CPU)
sense_tokens = projections(sense_feats)    # (9, 2048) JAX array
carry, r, fe = halo3_step(model, carry, text_tokens, sense_tokens, key)
```

Log line gains modality indicators: `[A][V]` when audio+vision live, `[ ][ ]` when blind/deaf.

### `memory/schema.py`
Episode gains two optional fields storing pre-projection features:
```python
audio_features: np.ndarray | None  # (8, 768) pre-projection
vision_features: np.ndarray | None  # (512,) pre-projection
```
Pre-projection features are stored so dream replay can re-project through whatever the projections have learned by dream time — Avatar reinterprets old memories with evolved understanding.

---

## Error Handling & Graceful Degradation

**Capture agent not running**: `meta.json` missing or stale → `sense_buffer` returns `SenseFeatures(None, None)` → `projections()` returns all-zeros `(9, 2048)` → Avatar runs as text-only. No crash, no special case in tick loop.

**Encoder load failure**: If Wav2Vec2 or CLIP fail to load at container startup, log warning and permanently return `None`. Avatar degrades to text-only. Senses are additive, not a hard dependency.

**Corrupt audio/image file**: Wrapped in try/except in AudioSense/VisionSense — returns `None` on any read or inference error.

---

## Dockerfile Changes

Add to pip installs:
```
torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, ~700MB
transformers>=4.40.0
soundfile
Pillow
```

Models download on first container start and cache in `data/model_cache/` (covered by existing volume mount).

No new ports. No new services.

---

## File Summary

```
capture_agent/
  capture_agent.py          # Windows host — mic + camera -> data/senses/
  requirements.txt          # sounddevice, opencv-python, numpy

halo3/senses/
  __init__.py
  audio_sense.py            # Wav2Vec2-base, CPU, output (8, 768)
  vision_sense.py           # CLIP ViT-B/32, CPU, output (512,)
  sense_buffer.py           # reads shared volume, freshness check, SenseFeatures
  projections.py            # Equinox SenseProjections, JAX/GPU, trained per-tick

halo3/config.py             # +4 sense fields (n_audio_tokens, n_vision_tokens, dims)
halo3/model.py              # SenseProjections on Halo3Model, halo3_step +sense_tokens arg
halo3/main.py               # sense_buffer + projections wired into tick loop
halo3/memory/schema.py      # Episode +audio_features, +vision_features
Dockerfile                  # +torch-cpu, transformers, soundfile, Pillow
```

---

## What Does NOT Change

- Lorentz embedding, backbone, Kuramoto, Hamiltonian — unchanged
- Per-tick Adam optimizer — unchanged (just more params in filter_spec)
- Dream cycle phases 1-4 — unchanged (episodes now carry sense features for replay)
- Chat server — unchanged
- PFC / Qwen3 — unchanged
- All existing drives, emotions, consciousness modules — unchanged

Drives and emotions are naturally affected by senses through r (order parameter). When Avatar hears something it cannot pattern-match, r drops → anxiety. When it sees something coherent, r rises → curiosity or satisfaction. No special wiring needed — this emerges from the unified latent space.
