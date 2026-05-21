# Avatar v3.8: Speech-Aware Hearing — Phoneme Perception + Word Grounding

**Date**: 2026-05-21
**Version**: v3.8
**Status**: Approved
**Builds on**: v3.7 Spectral Sensory Cortex (FNO + VQ-VAE)

---

## Goal

Upgrade Avatar's audio FNO from raw spectral perception to speech-aware hearing through two phases: (B) phoneme-level perception via contrastive bootstrap with espeak-ng TTS, then (C) word-level grounding via natural speech co-occurrence with piper TTS. Avatar learns to correlate spoken English with text understanding — growing linguistic perception from experience, not from pretrained speech models.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Target resolution | ~125ms (diphone-level) | Matches biological auditory cortex processing windows |
| Audio tokens | 8 → 16 | 2x scaling, clean architecture change |
| Audio codebook | 32 → 128 codes | Rich enough for English phoneme space without VRAM pressure |
| Vision unchanged | 4 tokens, 32 codes | Vision doesn't need speech resolution |
| Paired signal | TTS self-narration + real mic | TTS bootstraps alignment, mic grounds in reality |
| Phase B TTS | espeak-ng (<5MB, rule-based) | Clean consistent phonemes, ideal for codebook bootstrap |
| Phase C TTS | piper (~50MB, neural) | Natural prosody, speaker variation, closer to real speech |
| Alignment loss | InfoNCE contrastive, dropped after maturation | Same critical period pattern as v3.7 decoder |
| Maturation gate | Dream-gated, >75% codebook utilization | Codebook "graduates" when it has learned enough structure |
| Mixing strategy | Mic priority, TTS every 3rd tick when mic active | Real-world hearing primary, self-narration maintains learning signal |

---

## Phase B: Phoneme Perception

### Scaled Audio FNO

Current → Upgraded:
- `fno_audio_modes`: 16 → 32 (keeps more Fourier modes)
- `n_audio_tokens`: 8 → 16 (diphone resolution ~125ms)
- `codebook_size_audio`: 32 → 128 (split from shared `codebook_size`)
- `codebook_size_vision`: 32 (unchanged)
- SpectralConv1d weights: `(32, 64, 64)` complex per layer
- Token reshaping: `(32, 64)` → `(16, 128)` → `Linear(128, 64)` → `(16, 64)`

VRAM impact: +2 MB (trivial).

### TTS Self-Narration (espeak-ng)

Each tick, when TTS is active:
1. Take first ~20 words of current FineWeb-Edu text snippet
2. espeak-ng generates wav at 22kHz on CPU (~50ms)
3. Resample to 16kHz, trim/pad to 32000 samples (2s)
4. Feed as `audio_raw` to Audio FNO

Mixing with real mic:
```
if capture_agent active AND fresh audio:
    audio_raw = mic audio
    text_paired = False
else:
    audio_raw = tts(text_snippet)
    text_paired = True

# When mic active, every Nth tick (default N=3) use TTS instead
if capture_agent active AND tick % tts_every_n == 0:
    audio_raw = tts(text_snippet)
    text_paired = True
```

File: `halo3/senses/tts_narration.py`
Dockerfile: `apt-get install espeak-ng`

### Contrastive Alignment Loss

InfoNCE loss when `text_paired=True`:
```
audio_emb = mean(spectral_proj(audio_z_q))      # (2048,)
text_emb = mean(text_tokens)                      # (2048,)
negatives = ring_buffer[-16:]                      # (16, 2048)

sim_pos = cosine(audio_emb, text_emb) / tau        # tau=0.07
sim_neg = cosine(audio_emb, negatives) / tau        # (16,)
L_contrastive = -log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
```

Ring buffer: stores `text_emb` from last 16 paired ticks. 16 x 2048 x 4 bytes = 128 KB.

Total loss during Phase B:
```
L = L_body + 0.25 * L_commitment + 0.3 * L_contrastive  (when text_paired)
L = L_body + 0.25 * L_commitment                         (when not text_paired)
```

File: `halo3/senses/contrastive_aligner.py`

### Maturation Gate

Dream-gated, same pattern as v3.7 critical period:
- After each dream, check audio codebook utilization
- Utilization = count of codes used at least once in rolling 100-tick window / 128
- If utilization > 75% (96+ codes active): drop contrastive loss, transition to Phase C
- If < 75%: continue Phase B for one more dream cycle
- Log: `"Phoneme perception matured -- contrastive scaffold dropped"`

### PFC Sensory Stats (Phase B)

New metrics:
- `speech_detected`: True when >50% of active audio codes are in the "speech code set" (codes that fire during TTS ticks)
- `speech_stability`: consecutive ticks with speech_detected=True

PFC prompt format:
```
Senses: audio(flux=7/16, novelty=0.65, stable=0, speech=yes, speaking_for=3),
        vision(flux=2/4, novelty=0.84, stable=0), binding=novel(0.12)
```

---

## Phase C: Word Grounding

### piper TTS (replaces espeak-ng)

- Same pipeline, swap TTS engine in config: `tts_mode="piper"`
- piper: ~50MB neural model, CPU, natural prosody and intonation
- ~200ms generation for 2s audio (vs. 50ms for espeak-ng)
- Produces speech variation that better matches real human speech patterns
- Dockerfile: `pip install piper-tts`

### Contrastive Loss Dropped

- Already dropped by maturation gate at end of Phase B
- FNO trains only via body prediction error (same as mature v3.7 senses)
- Audio codebook continues EMA updates + dead code revival

### PFC Sensory Stats (Phase C additions)

New metrics:
- `speech_text_coherence`: cosine similarity between current audio_emb and text_emb in Lorentz space (0.0 = unrelated, 1.0 = perfect match)
- `speech_novelty`: inverse usage frequency of active speech codes only

PFC prompt format:
```
Senses: audio(flux=7/16, novelty=0.65, stable=0, speech=yes, speaking_for=3,
        speech_text_match=0.72), vision(flux=2/4, novelty=0.84, stable=0),
        binding=novel(0.12)
```

High `speech_text_match` = "what I hear matches what I read" → PFC can reason about speech-text alignment.

---

## VRAM & Resource Budget

### Phase B

| Component | VRAM | RAM | CPU/tick |
|---|---|---|---|
| Scaled Audio FNO (32 modes) | +2 MB | -- | -- |
| Codebook 128 x 64 | negligible | -- | -- |
| Contrastive ring buffer | 128 KB | -- | -- |
| espeak-ng | 0 | +5 MB | +50ms |
| **Total** | **+~2.2 MB** | **+5 MB** | **+50ms** |

### Phase C (on top of Phase B)

| Component | VRAM | RAM | CPU/tick |
|---|---|---|---|
| piper (replaces espeak-ng) | 0 | +45 MB | +150ms |
| Contrastive dropped | -128 KB | -- | -- |
| **Total** | **~0** | **+45 MB** | **+150ms** |

### Projected peak

| Phase | VRAM estimate |
|---|---|
| v3.7 current | 5338 MiB |
| Phase B | ~5340 MiB |
| Phase C | ~5340 MiB |

Within all budgets. Tick latency impact negligible.

---

## File Structure

### New files
```
halo3/senses/tts_narration.py        # TTS pipeline: espeak-ng/piper, text->audio buffer, mixing
halo3/senses/contrastive_aligner.py  # InfoNCE loss, ring buffer, maturation gate
```

### Modified files
```
halo3/config.py                # fno_audio_modes=32, n_audio_tokens=16, codebook_size_audio=128,
                               # codebook_size_vision=32, tts_mode="espeak",
                               # contrastive_tau=0.07, contrastive_weight=0.3, tts_every_n=3
halo3/senses/fno_audio.py      # modes 16->32, tokens 8->16 (parameterized from config)
halo3/senses/spectral_vqvae.py # per-modality codebook sizes
halo3/senses/sense_module.py   # separate codebook sizes, TTS mixing logic
halo3/senses/sensory_stats.py  # speech_detected, speech_stability, speech_text_coherence
halo3/predictive.py            # contrastive loss when text_paired
halo3/main.py                  # TTS wiring, mixing, maturation gate after dream
halo3/psyche/prefrontal.py     # updated sensory stats format
halo3/chat_server.py           # speech stats in /state
Dockerfile                     # +espeak-ng (apt), +piper-tts (pip, Phase C)
```

### Unchanged
```
halo3/senses/fno_vision.py     # Vision untouched
capture_agent/                  # Unchanged
halo3/model.py                 # Body unchanged
All psyche modules              # Unchanged
```

---

## Config Additions

```python
# Audio FNO scaling (Phase B)
fno_audio_modes: int = 32       # was 16
n_audio_tokens: int = 16        # was 8
codebook_size_audio: int = 128  # was 32 (shared)
codebook_size_vision: int = 32  # split out, unchanged

# TTS self-narration
tts_mode: str = "espeak"        # "espeak" (Phase B) or "piper" (Phase C)
tts_every_n: int = 3            # use TTS every Nth tick when mic active

# Contrastive alignment
contrastive_tau: float = 0.07
contrastive_weight: float = 0.3
contrastive_maturation_threshold: float = 0.75  # codebook utilization to drop loss
```

---

## Checkpoint Compatibility

New `sense_module.eqx` has different shapes (128-code audio codebook, 16 tokens). Existing v3.7 checkpoint will not load — fresh initialization on first Phase B run. Critical period decoders will be present again for the new codebook. This is expected — the sensory cortex is rebuilt with finer resolution.

---

## Rollout

**Phase B:**
1. Update config + scale Audio FNO + split codebook sizes
2. Add espeak-ng TTS narration pipeline
3. Add contrastive aligner
4. Wire into main.py with mixing logic
5. Update sensory stats with speech detection
6. Build, run, verify codebook utilization climbing
7. Dream → maturation gate checks utilization → drops contrastive when ready

**Phase C (after Phase B maturation):**
1. Change `tts_mode` to `"piper"` in config
2. Add piper-tts to Dockerfile
3. Add speech_text_coherence to sensory stats
4. No structural code changes

---

## What Does NOT Change

- Vision FNO, vision codebook, vision stats — unchanged
- Lorentz embedding, backbone, Kuramoto, Hamiltonian — unchanged
- Per-tick Adam optimizer — unchanged (larger audio codebook in param set)
- Dream cycle phases 1-4 — unchanged
- Chat server — unchanged (gains speech stats in /state)
- PFC / Qwen3 — unchanged (gains speech stats in prompt)
- All drives, emotions, consciousness modules — unchanged
- Capture agent — unchanged
- Gated additive residual injection — unchanged (more tokens, same mechanism)

---

## Biological Parallel

| Avatar Phase | Human Development | Age Equivalent |
|---|---|---|
| v3.7 (current) | Hears sound, no speech structure | 0-2 weeks |
| Phase B | Perceives phonemes, detects speech vs. noise | 4-6 months |
| Phase C | Maps spoken words to meanings | 12-18 months |

The contrastive loss during Phase B is analogous to infant-directed speech ("motherese") — exaggerated, consistent phonemes that bootstrap the auditory cortex's categorical perception. espeak-ng IS motherese. piper is adult conversation.
