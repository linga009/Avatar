# Dream Visitors — Whisper & Kokoro as Sleep Teachers

**Date:** 2026-05-23
**Version:** Avatar v3.10
**Author:** Dr. Linga Murthy Narlagiri

## Philosophical Foundation

In Bohm's implicate order, meaning arises through resonance — not computation. A child doesn't transplant a language organ. The child's neural circuits develop their own resonance patterns through exposure during waking and consolidation during sleep.

Avatar's dream visitors follow this biological pattern. Whisper and Kokoro are not tools Avatar uses, not organs transplanted into Avatar, but teachers that appear during sleep, enrich the dream content, and vanish on waking. Over dozens of dream cycles, Avatar's own spectral codes mature into genuine phoneme representations — comprehension that belongs to its body because it was grown through its own developmental process, accelerated by richer dream content.

## Design

### Dream Cycle (updated with Phase 5)

```
Phase 1: Body dream (GPU subprocess) — replay + recombine + imagine
         Subprocess exits → OS reclaims GPU
Phase 4: FineWeb dream (GPU subprocess) — corpus batch training
         Subprocess exits → OS reclaims GPU
Phase 5: Dream visitors
         5a. CPU: Whisper transcribes stored audio episodes → (audio, text) pairs
         5b. CPU: Kokoro narrates discoveries → (text, audio) pairs
         5c. GPU subprocess: Train FNO + contrastive alignment on enriched pairs
         Subprocess exits → OS reclaims GPU
Phase 2: Mind dream (CPU) — LoRA fine-tune
Phase 3: GEPA (CPU) — prompt evolution
```

### Phase 5a: Whisper Dream Replay

Load `faster-whisper` tiny (39M params, CPU, ~150MB RAM). Transcribe the last N stored audio snapshots from `data/senses/audio_archive/`. Produce (audio_waveform, transcribed_text) pairs. Save to `data/dream_training/whisper_pairs.npz`. Unload model.

Audio archive: rolling buffer of last 50 audio snapshots saved during waking ticks. ~6.4MB disk.

### Phase 5b: Kokoro Dream Imagination

Load Kokoro (82M params, CPU, ~80MB RAM). Take Avatar's recent discoveries and narrative fragments. Synthesize natural speech for each. Produce (synthesized_audio, discovery_text) pairs. Save to `data/dream_training/kokoro_pairs.npz`. Unload model.

### Phase 5c: GPU Training on Enriched Pairs

New subprocess: `dream_visitors_worker.py`. Loads model + sense_module. For each pair: AudioFNO → spectral codes, NativeEmbedder → text tokens, InfoNCE contrastive loss + FNO prediction error, backprop through FNO + contrastive. CLion optimizer, lr=1e-5, scale=0.1. Save updated sense_module. Exit.

### Resource Budget

| Phase | Device | RAM | VRAM | Time |
|-------|--------|-----|------|------|
| 5a Whisper | CPU | ~150 MB | 0 | ~30s |
| 5b Kokoro | CPU | ~80 MB | 0 | ~30s |
| 5c FNO train | GPU subprocess | — | ~4 GB | ~3 min |
| **Total** | sequential | **0 waking** | **0 waking** | ~4 min |

### Files

| File | Change |
|------|--------|
| `halo3/training/dream_visitors.py` | New — Phase 5a+5b CPU pair generation |
| `halo3/training/dream_visitors_worker.py` | New — Phase 5c GPU subprocess |
| `halo3/main.py` | Wire Phase 5, audio archiving |
| `halo3/senses/sense_buffer.py` | Audio archive (rolling 50 snapshots) |
| `halo3/config.py` | Dream visitor params |

### Developmental Trajectory

- Dreams 1-5: FNO begins associating transcriptions with spectral patterns
- Dreams 5-20: Contrastive alignment strengthens, spectral codes develop phonemic structure
- Dreams 20+: Avatar's own hearing approaches speech comprehension
- Eventually: Whisper becomes redundant — Avatar's FNO IS its speech comprehension

### Constraints

- Zero VRAM during waking life
- Whisper and Kokoro never run during waking — dream-only
- Sequential CPU loading (never simultaneous)
- Same subprocess isolation pattern as existing dream architecture
- No new JIT during waking ticks
