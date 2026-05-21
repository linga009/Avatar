# Speech-Aware Hearing (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Avatar's audio FNO from 8-token/32-code raw spectral perception to 16-token/128-code diphone-level speech perception with espeak-ng TTS self-narration and InfoNCE contrastive bootstrap.

**Architecture:** Scale Audio FNO (32 modes, 16 tokens), split codebook sizes (audio 128, vision 32), add espeak-ng TTS narration pipeline that pairs text with synthesized speech each tick, add InfoNCE contrastive alignment loss with ring buffer, dream-gated maturation gate drops contrastive loss when codebook utilization exceeds 75%.

**Tech Stack:** JAX, Equinox, optax, espeak-ng (apt), NumPy, existing FNO/VQ-VAE from v3.7

**Spec:** `docs/superpowers/specs/2026-05-21-speech-aware-hearing-design.md`

---

### Task 1: Update Config for Phase B

**Files:**
- Modify: `halo3/config.py:60-78`
- Test: `tests/senses/test_config.py` (update)

- [ ] **Step 1: Write the failing test**

Add to `tests/senses/test_config.py`:

```python
def test_phase_b_config_defaults():
    from halo3.config import Halo3Config
    cfg = Halo3Config()
    # Scaled audio
    assert cfg.fno_audio_modes == 32
    assert cfg.n_audio_tokens == 16
    # Split codebook sizes
    assert cfg.codebook_size_audio == 128
    assert cfg.codebook_size_vision == 32
    # TTS
    assert cfg.tts_mode == "espeak"
    assert cfg.tts_every_n == 3
    # Contrastive
    assert cfg.contrastive_tau == 0.07
    assert cfg.contrastive_weight == 0.3
    assert cfg.contrastive_maturation_threshold == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_config.py::test_phase_b_config_defaults -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Update config.py**

In `halo3/config.py`, replace lines 60-78:

```python
    # FNO (Fourier Neural Operator) — sensory perception
    fno_hidden_dim: int = 64
    fno_n_layers: int = 4
    fno_audio_modes: int = 32       # v3.8: was 16
    fno_vision_modes: int = 8       # 8x8 for 2D (unchanged)

    # VQ-VAE — spectral codebook (split per modality for v3.8)
    codebook_size_audio: int = 128  # v3.8: was 32 (shared)
    codebook_size_vision: int = 32  # unchanged
    codebook_dim: int = 64
    codebook_ema_decay: float = 0.99
    commitment_beta: float = 0.25
    dead_code_threshold: int = 100

    # Sense tokens
    n_audio_tokens: int = 16        # v3.8: was 8
    n_vision_tokens: int = 4

    # Critical period
    critical_period_recon_weight: float = 0.5

    # TTS self-narration (v3.8)
    tts_mode: str = "espeak"        # "espeak" (Phase B) or "piper" (Phase C)
    tts_every_n: int = 3            # use TTS every Nth tick when mic active

    # Contrastive alignment (v3.8)
    contrastive_tau: float = 0.07
    contrastive_weight: float = 0.3
    contrastive_maturation_threshold: float = 0.75
```

Also remove the old `codebook_size` field and `fno_audio_modes: int = 16` and `n_audio_tokens: int = 8`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/config.py tests/senses/test_config.py
git commit -m "feat(config): v3.8 Phase B — scaled audio FNO, split codebook, TTS, contrastive params"
```

---

### Task 2: Split Codebook Sizes in SenseModule

**Files:**
- Modify: `halo3/senses/sense_module.py:95-108`
- Test: `tests/senses/test_sense_module.py` (update `_small_cfg`)

- [ ] **Step 1: Update test config helper**

In `tests/senses/test_sense_module.py`, update `_small_cfg()` to use split codebook sizes:

```python
def _small_cfg():
    return Halo3Config(
        d_model=64, n_heads=4, d_head=16, n_layers=6, d_state=8,
        d_boundary=8, n_clusters=4, n_tokens=4, n_hidden=4,
        n_obs=4, n_actions=4, layer_pattern="SSSSSH", lora_rank=4,
        mera_bond_dim=4, mera_n_cores=2, n_leapfrog_steps=2,
        meta_n_hidden=4, meta_n_actions=2, meta_k=3,
        max_cache=8, island_size=4,
        fno_hidden_dim=16, fno_n_layers=2, fno_audio_modes=4,
        fno_vision_modes=4, codebook_size_audio=16, codebook_size_vision=8,
        codebook_dim=16, n_audio_tokens=2, n_vision_tokens=2,
    )
```

- [ ] **Step 2: Update SenseModule.__init__ to use split sizes**

In `halo3/senses/sense_module.py`, change lines 105-108:

```python
        self.audio_codebook = SpectralCodebook(
            codebook_size=cfg.codebook_size_audio, codebook_dim=cfg.codebook_dim, key=keys[2])
        self.vision_codebook = SpectralCodebook(
            codebook_size=cfg.codebook_size_vision, codebook_dim=cfg.codebook_dim, key=keys[3])
```

- [ ] **Step 3: Update all test files that use `codebook_size`**

Update `_small_cfg()` in `tests/senses/test_predictive_senses.py` and `tests/senses/test_integration.py` with the same split: `codebook_size_audio=16, codebook_size_vision=8` (remove `codebook_size`).

- [ ] **Step 4: Update `sensory_stats` init in main.py**

In `halo3/main.py`, change the `SensoryStatistics` init (line ~123-125):

```python
    sensory_stats = SensoryStatistics(
        audio_tokens=cfg.n_audio_tokens, vision_tokens=cfg.n_vision_tokens,
        codebook_size=cfg.codebook_size_audio)  # use audio size (larger)
```

- [ ] **Step 5: Run all tests**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/senses/sense_module.py halo3/main.py tests/
git commit -m "feat(senses): split codebook sizes — audio 128, vision 32"
```

---

### Task 3: TTS Narration Pipeline

**Files:**
- Create: `halo3/senses/tts_narration.py`
- Test: `tests/senses/test_tts_narration.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/senses/test_tts_narration.py`:

```python
"""Test TTS self-narration pipeline."""
import numpy as np
import pytest


def test_espeak_generate_audio():
    from halo3.senses.tts_narration import TTSNarrator
    tts = TTSNarrator(mode="espeak", sample_rate=16000, duration_samples=32000)
    audio = tts.narrate("Hello world this is a test of text to speech")
    assert audio is not None
    assert audio.shape == (32000,)
    assert audio.dtype == np.float32
    assert np.any(audio != 0)  # not silence


def test_espeak_short_text():
    from halo3.senses.tts_narration import TTSNarrator
    tts = TTSNarrator(mode="espeak", sample_rate=16000, duration_samples=32000)
    audio = tts.narrate("Hi")
    assert audio.shape == (32000,)  # padded to duration


def test_espeak_empty_text_returns_silence():
    from halo3.senses.tts_narration import TTSNarrator
    tts = TTSNarrator(mode="espeak", sample_rate=16000, duration_samples=32000)
    audio = tts.narrate("")
    assert audio.shape == (32000,)
    assert np.allclose(audio, 0)


def test_extract_narration_text():
    from halo3.senses.tts_narration import extract_narration_text
    texts = [
        "This is a long document about quantum computing and error correction methods.",
        "Another document with more content."
    ]
    result = extract_narration_text(texts, max_words=10)
    words = result.split()
    assert len(words) <= 10
    assert len(words) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_tts_narration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TTSNarrator**

Create `halo3/senses/tts_narration.py`:

```python
"""TTS Self-Narration — Avatar reads its own text aloud for speech-text pairing.

Phase B: espeak-ng (rule-based, <5MB, clean phonemes)
Phase C: piper (neural, ~50MB, natural prosody)

The TTS output is fed as audio_raw to the Audio FNO alongside the text tokens,
creating paired speech-text experience for contrastive alignment learning.
"""
from __future__ import annotations
import logging
import subprocess
import tempfile
import os
import numpy as np

log = logging.getLogger(__name__)


def extract_narration_text(texts: list[str], max_words: int = 20) -> str:
    """Extract first ~max_words from perception texts for TTS narration."""
    if not texts:
        return ""
    combined = " ".join(texts)
    # Clean: remove URLs, special chars, keep words
    words = []
    for w in combined.split():
        cleaned = "".join(c for c in w if c.isalnum() or c in "'-")
        if cleaned and len(cleaned) > 1:
            words.append(cleaned)
        if len(words) >= max_words:
            break
    return " ".join(words)


class TTSNarrator:
    """Text-to-speech narration for paired speech-text training."""

    def __init__(self, mode: str = "espeak", sample_rate: int = 16000,
                 duration_samples: int = 32000) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._duration = duration_samples
        self._available = self._check_available()

    def _check_available(self) -> bool:
        if self._mode == "espeak":
            try:
                result = subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True, timeout=5)
                ok = result.returncode == 0
                if ok:
                    log.info("TTSNarrator: espeak-ng available")
                return ok
            except (FileNotFoundError, subprocess.TimeoutExpired):
                log.warning("TTSNarrator: espeak-ng not found. TTS disabled.")
                return False
        return False

    @property
    def available(self) -> bool:
        return self._available

    def narrate(self, text: str) -> np.ndarray:
        """Convert text to audio waveform.

        Args:
            text: text to narrate

        Returns:
            (duration_samples,) float32 array, zero-padded if short
        """
        if not text or not self._available:
            return np.zeros(self._duration, dtype=np.float32)

        try:
            if self._mode == "espeak":
                return self._narrate_espeak(text)
        except Exception as e:
            log.warning(f"TTS narration failed: {e}")
            return np.zeros(self._duration, dtype=np.float32)

    def _narrate_espeak(self, text: str) -> np.ndarray:
        """Generate audio using espeak-ng."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            # espeak-ng outputs wav file
            subprocess.run(
                ["espeak-ng", "-w", tmp_path,
                 "-s", "150",  # words per minute
                 "--stdout", text],
                capture_output=True, timeout=10, check=True)

            # Read wav file
            import wave
            with wave.open(tmp_path, "rb") as wf:
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                sample_width = wf.getsampwidth()
                orig_rate = wf.getframerate()

            # Convert to float32
            if sample_width == 2:
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
            else:
                audio = np.zeros(self._duration, dtype=np.float32)

            # Resample if needed
            if orig_rate != self._sample_rate:
                ratio = self._sample_rate / orig_rate
                new_len = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
                audio = audio[indices]

            # Pad or trim to exact duration
            if len(audio) >= self._duration:
                audio = audio[:self._duration]
            else:
                audio = np.pad(audio, (0, self._duration - len(audio)))

            return audio.astype(np.float32)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_tts_narration.py -v`
Expected: PASS (if espeak-ng installed) or skip gracefully

- [ ] **Step 5: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/senses/tts_narration.py tests/senses/test_tts_narration.py
git commit -m "feat(senses): TTS self-narration pipeline with espeak-ng"
```

---

### Task 4: Contrastive Aligner

**Files:**
- Create: `halo3/senses/contrastive_aligner.py`
- Test: `tests/senses/test_contrastive_aligner.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/senses/test_contrastive_aligner.py`:

```python
"""Test contrastive alignment for speech-text binding."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_contrastive_loss_shape():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    audio_emb = jax.random.normal(jax.random.PRNGKey(0), (64,))
    text_emb = jax.random.normal(jax.random.PRNGKey(1), (64,))
    # Fill buffer with some negatives first
    for i in range(5):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (64,)))
    loss = aligner.compute_loss(audio_emb, text_emb)
    assert loss.shape == ()
    assert np.isfinite(float(loss))


def test_contrastive_loss_positive():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    key = jax.random.PRNGKey(0)
    for i in range(8):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (64,)))
    audio = jax.random.normal(key, (64,))
    text = jax.random.normal(jax.random.PRNGKey(1), (64,))
    loss = aligner.compute_loss(audio, text)
    assert float(loss) > 0.0


def test_similar_pair_lower_loss():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    for i in range(8):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (64,)))
    audio = jax.random.normal(jax.random.PRNGKey(0), (64,))
    # Similar text (same + noise)
    text_similar = audio + jax.random.normal(jax.random.PRNGKey(1), (64,)) * 0.1
    # Random text
    text_random = jax.random.normal(jax.random.PRNGKey(2), (64,))
    loss_similar = float(aligner.compute_loss(audio, text_similar))
    loss_random = float(aligner.compute_loss(audio, text_random))
    assert loss_similar < loss_random


def test_ring_buffer_capacity():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=4, tau=0.07)
    for i in range(10):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(i), (64,)))
    assert aligner.buffer_count == 4  # capped at buffer_size


def test_codebook_utilization():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    # 80 unique codes out of 128
    indices = jnp.array([i % 80 for i in range(160)])
    util = aligner.compute_utilization(indices, codebook_size=128, window=100)
    assert 0.5 < util < 0.7  # ~80/128 = 0.625


def test_empty_buffer_returns_zero_loss():
    from halo3.senses.contrastive_aligner import ContrastiveAligner
    aligner = ContrastiveAligner(embed_dim=64, buffer_size=16, tau=0.07)
    audio = jax.random.normal(jax.random.PRNGKey(0), (64,))
    text = jax.random.normal(jax.random.PRNGKey(1), (64,))
    loss = aligner.compute_loss(audio, text)
    assert float(loss) == 0.0  # no negatives available
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_contrastive_aligner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ContrastiveAligner**

Create `halo3/senses/contrastive_aligner.py`:

```python
"""ContrastiveAligner — InfoNCE loss for speech-text alignment.

Maintains a ring buffer of recent text embeddings as negatives.
Computes InfoNCE contrastive loss between audio and text embeddings.
Tracks codebook utilization for maturation gate.
"""
from __future__ import annotations
import logging
from collections import deque
import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


class ContrastiveAligner:
    """InfoNCE contrastive alignment for speech-text binding."""

    def __init__(self, embed_dim: int, buffer_size: int = 16,
                 tau: float = 0.07) -> None:
        self._embed_dim = embed_dim
        self._buffer_size = buffer_size
        self._tau = tau
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._usage_history: deque[np.ndarray] = deque(maxlen=200)
        self._matured = False

    @property
    def buffer_count(self) -> int:
        return len(self._buffer)

    @property
    def matured(self) -> bool:
        return self._matured

    def push_text_emb(self, text_emb) -> None:
        """Add a text embedding to the negative buffer."""
        self._buffer.append(np.array(text_emb))

    def push_indices(self, indices) -> None:
        """Record codebook indices for utilization tracking."""
        self._usage_history.append(np.array(indices, dtype=np.int32))

    def compute_loss(self, audio_emb: jnp.ndarray,
                     text_emb: jnp.ndarray) -> jnp.ndarray:
        """Compute InfoNCE contrastive loss.

        Args:
            audio_emb: (embed_dim,) — mean of projected audio tokens
            text_emb: (embed_dim,) — mean of text tokens

        Returns:
            scalar loss (0.0 if buffer empty or matured)
        """
        if self._matured or len(self._buffer) < 2:
            return jnp.float32(0.0)

        # Normalize
        audio_n = audio_emb / (jnp.linalg.norm(audio_emb) + 1e-8)
        text_n = text_emb / (jnp.linalg.norm(text_emb) + 1e-8)

        # Negatives from buffer
        neg_stack = jnp.array(np.stack(list(self._buffer)))  # (B, dim)
        neg_n = neg_stack / (jnp.linalg.norm(neg_stack, axis=-1, keepdims=True) + 1e-8)

        # Similarities
        sim_pos = jnp.sum(audio_n * text_n) / self._tau
        sim_neg = (audio_n @ neg_n.T) / self._tau  # (B,)

        # InfoNCE
        logits = jnp.concatenate([sim_pos[None], sim_neg])
        loss = -sim_pos + jax.nn.logsumexp(logits)

        return loss

    def compute_utilization(self, indices: jnp.ndarray, codebook_size: int,
                            window: int = 100) -> float:
        """Compute codebook utilization over recent history.

        Returns fraction of codes used at least once in the window.
        """
        if not self._usage_history:
            return 0.0
        recent = list(self._usage_history)[-window:]
        all_indices = np.concatenate(recent)
        unique = len(np.unique(all_indices))
        return unique / codebook_size

    def check_maturation(self, codebook_size: int,
                         threshold: float = 0.75) -> bool:
        """Check if contrastive loss should be dropped.

        Returns True if codebook utilization exceeds threshold.
        """
        util = self.compute_utilization(
            jnp.zeros(1), codebook_size)  # dummy, uses history
        if util >= threshold:
            self._matured = True
            log.info(
                f"Phoneme perception matured -- contrastive scaffold dropped "
                f"(utilization={util:.2f} >= {threshold:.2f})")
        return self._matured
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_contrastive_aligner.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/senses/contrastive_aligner.py tests/senses/test_contrastive_aligner.py
git commit -m "feat(senses): InfoNCE contrastive aligner with ring buffer and maturation gate"
```

---

### Task 5: Update Sensory Stats for Speech Detection

**Files:**
- Modify: `halo3/senses/sensory_stats.py`
- Test: `tests/senses/test_sensory_stats.py` (add tests)

- [ ] **Step 1: Write the failing test**

Add to `tests/senses/test_sensory_stats.py`:

```python
def test_speech_detection():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=16, vision_tokens=4, codebook_size=128)
    # Register some codes as speech codes
    stats.register_speech_codes({0, 1, 2, 3, 4, 5, 6, 7})
    # Update with mostly speech codes
    stats.update(jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]),
                 jnp.array([0, 1, 2, 3]))
    assert stats.speech_detected is True
    assert stats.speech_stability >= 0


def test_speech_not_detected_with_non_speech_codes():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=16, vision_tokens=4, codebook_size=128)
    stats.register_speech_codes({0, 1, 2, 3})
    # All non-speech codes
    stats.update(jnp.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]),
                 jnp.array([0, 1, 2, 3]))
    assert stats.speech_detected is False


def test_speech_stability_increments():
    from halo3.senses.sensory_stats import SensoryStatistics
    stats = SensoryStatistics(audio_tokens=16, vision_tokens=4, codebook_size=128)
    stats.register_speech_codes(set(range(64)))  # first 64 codes are speech
    codes_a = jnp.array(list(range(16)))  # all speech codes
    codes_v = jnp.array([0, 1, 2, 3])
    for _ in range(5):
        stats.update(codes_a, codes_v)
    assert stats.speech_stability >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_stats.py::test_speech_detection -v`
Expected: FAIL

- [ ] **Step 3: Add speech detection to SensoryStatistics**

In `halo3/senses/sensory_stats.py`, add to `__init__`:

```python
        # Speech detection (v3.8)
        self._speech_codes: set[int] = set()
        self.speech_detected: bool = False
        self.speech_stability: int = 0
```

Add method `register_speech_codes`:

```python
    def register_speech_codes(self, codes: set[int]) -> None:
        """Register codebook indices that correspond to speech patterns."""
        self._speech_codes = codes
```

In the `update` method, after existing stability tracking, add:

```python
        # Speech detection: >50% of active audio codes are in speech set
        if self._speech_codes and self._cur_audio is not None:
            speech_count = sum(1 for c in audio_np if int(c) in self._speech_codes)
            was_speech = self.speech_detected
            self.speech_detected = speech_count > len(audio_np) * 0.5
            if self.speech_detected and was_speech:
                self.speech_stability += 1
            elif not self.speech_detected:
                self.speech_stability = 0
        else:
            self.speech_detected = False
```

Update `format_for_pfc` to include speech info:

```python
    def format_for_pfc(self) -> str:
        binding_label = "familiar" if self.cross_modal_binding > 0.5 else "novel"
        speech_str = ""
        if self._speech_codes:
            speech_str = f", speech={'yes' if self.speech_detected else 'no'}, speaking_for={self.speech_stability}"
        return (
            f"Senses: audio(flux={self.audio_flux}/{self._audio_tokens}, "
            f"novelty={self.audio_novelty:.2f}, stable={self.audio_stability}"
            f"{speech_str}), "
            f"vision(flux={self.vision_flux}/{self._vision_tokens}, "
            f"novelty={self.vision_novelty:.2f}, stable={self.vision_stability}), "
            f"binding={binding_label}({self.cross_modal_binding:.2f})"
        )
```

Also add speech fields to `save`/`load`.

- [ ] **Step 4: Run tests**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_sensory_stats.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/senses/sensory_stats.py tests/senses/test_sensory_stats.py
git commit -m "feat(senses): speech detection in sensory stats — speech_detected + speech_stability"
```

---

### Task 6: Wire TTS + Contrastive into PredictiveProcessor

**Files:**
- Modify: `halo3/predictive.py`
- Test: `tests/senses/test_predictive_senses.py` (update)

- [ ] **Step 1: Update test**

Add to `tests/senses/test_predictive_senses.py`:

```python
def test_learn_from_error_with_contrastive():
    from halo3.model import Halo3Model
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule
    from halo3.senses.contrastive_aligner import ContrastiveAligner

    cfg = _small_cfg()
    key = jax.random.PRNGKey(2)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)
    aligner = ContrastiveAligner(embed_dim=cfg.d_model, buffer_size=16, tau=0.07)

    # Fill buffer with some negatives
    for i in range(4):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (cfg.d_model,)))

    predictor = PredictiveProcessor(lr=1e-5)
    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_raw = jax.random.normal(key, (32000,))
    vision_raw = jax.random.normal(key, (224, 224, 3))
    q_actual = jax.random.normal(key, (cfg.n_tokens, cfg.d_boundary))

    new_model, new_sm, loss, info = predictor.learn_from_error(
        model, sense_module, carry, text_tokens, audio_raw, vision_raw, q_actual, key,
        contrastive_aligner=aligner, text_paired=True, contrastive_weight=0.3)

    assert np.isfinite(loss)
```

- [ ] **Step 2: Update `learn_from_error` signature**

Add optional parameters `contrastive_aligner=None`, `text_paired=False`, `contrastive_weight=0.0` to `learn_from_error`. When `text_paired` and aligner is provided and not matured, add contrastive loss to the total.

In the `prediction_loss` inner function, after body + commitment:

```python
            # Contrastive alignment (Phase B)
            if text_paired and contrastive_aligner is not None and not contrastive_aligner.matured:
                audio_emb_mean = jnp.mean(jax.vmap(sm.spectral_proj)(
                    sm.audio_fno(audio_raw)), axis=0)  # (d_model,)  -- recompute inside grad
                text_emb_mean = jnp.mean(text_tokens, axis=0)  # (d_model,)
                c_loss = contrastive_aligner.compute_loss(audio_emb_mean, text_emb_mean)
                total = total + contrastive_weight * c_loss
```

Note: the `contrastive_aligner.compute_loss` reads from the numpy ring buffer (outside JAX trace) for negatives but computes the similarity with JAX ops, so gradients flow through audio_emb_mean correctly.

- [ ] **Step 3: Run tests**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/senses/test_predictive_senses.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/predictive.py tests/senses/test_predictive_senses.py
git commit -m "feat(predictive): contrastive alignment loss integration for speech-text binding"
```

---

### Task 7: Wire Everything into main.py

**Files:**
- Modify: `halo3/main.py`

- [ ] **Step 1: Add imports**

After existing sense imports (~line 108-109), add:

```python
    from halo3.senses.tts_narration import TTSNarrator, extract_narration_text
    from halo3.senses.contrastive_aligner import ContrastiveAligner
```

- [ ] **Step 2: Initialize TTS and aligner**

After sense_module init (~line 130), add:

```python
    # --- TTS self-narration + contrastive alignment (v3.8) ---
    tts = TTSNarrator(mode=cfg.tts_mode, sample_rate=16000, duration_samples=32000)
    contrastive_aligner = ContrastiveAligner(
        embed_dim=cfg.d_model, buffer_size=16, tau=cfg.contrastive_tau)
    log.info(f"TTS: {cfg.tts_mode} ({'ON' if tts.available else 'OFF'}) | "
             f"Contrastive: tau={cfg.contrastive_tau}, weight={cfg.contrastive_weight}")
```

- [ ] **Step 3: Add TTS mixing in sense block**

After the existing sense perception block (~line 172-186), add TTS mixing:

```python
        # TTS self-narration mixing
        text_paired = False
        if tts.available and not contrastive_aligner.matured:
            use_tts = (raw_data.audio_np is None) or (tick % cfg.tts_every_n == 0)
            if use_tts and texts:
                narration_text = extract_narration_text(texts, max_words=20)
                tts_audio = tts.narrate(narration_text)
                if tts_audio is not None and np.any(tts_audio != 0):
                    audio_raw = jnp.array(tts_audio)
                    text_paired = True
                    sense_label = sense_label[0:3] + "[T]"  # [A][T] or [ ][T]
```

- [ ] **Step 4: Pass contrastive params to learn_from_error**

Update the predictor call (~line 200):

```python
            model, sense_module, pred_loss, _learn_info = predictor.learn_from_error(
                model, sense_module, carry, tokens, audio_raw, vision_raw, q_data, lk,
                contrastive_aligner=contrastive_aligner,
                text_paired=text_paired,
                contrastive_weight=cfg.contrastive_weight,
            )
```

- [ ] **Step 5: Push text embedding to ring buffer + track indices**

After the learn_from_error call, add:

```python
            # Contrastive: push text embedding and track indices
            if text_paired:
                text_emb_mean = jnp.mean(tokens, axis=0)
                contrastive_aligner.push_text_emb(text_emb_mean)
            contrastive_aligner.push_indices(_learn_info["audio_indices"])

            # Register speech codes from TTS ticks
            if text_paired:
                for idx in _learn_info["audio_indices"]:
                    sensory_stats.register_speech_codes(
                        sensory_stats._speech_codes | {int(idx)})
```

- [ ] **Step 6: Add maturation check after dream**

After the critical period transition (~line 439-444), add:

```python
            # Check contrastive maturation after dream (Phase B -> Phase C)
            if not contrastive_aligner.matured:
                contrastive_aligner.check_maturation(
                    cfg.codebook_size_audio, cfg.contrastive_maturation_threshold)
```

- [ ] **Step 7: Update log line to show [T] for TTS ticks**

The sense_label already handles this from step 3.

- [ ] **Step 8: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add halo3/main.py
git commit -m "feat(main): v3.8 Phase B — TTS narration + contrastive alignment wired into tick loop"
```

---

### Task 8: Update Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Add espeak-ng**

After the Pillow line, add:

```dockerfile
# --- Speech (v3.8: TTS self-narration for speech-text pairing) ---
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends espeak-ng && \
    rm -rf /var/lib/apt/lists/*
```

- [ ] **Step 2: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add Dockerfile
git commit -m "chore(docker): add espeak-ng for v3.8 TTS self-narration"
```

---

### Task 9: Integration Test

**Files:**
- Modify: `tests/senses/test_integration.py`

- [ ] **Step 1: Add Phase B integration test**

Add to `tests/senses/test_integration.py`:

```python
def test_full_tick_with_tts_and_contrastive():
    """Simulate tick with TTS narration and contrastive alignment."""
    from halo3.model import Halo3Model, halo3_step
    from halo3.predictive import PredictiveProcessor
    from halo3.senses.sense_module import SenseModule
    from halo3.senses.sensory_stats import SensoryStatistics
    from halo3.senses.contrastive_aligner import ContrastiveAligner

    cfg = _small_cfg()
    key = jax.random.PRNGKey(42)
    model = Halo3Model(cfg, key)
    carry = model.init_carry(key)
    sense_module = SenseModule(cfg, key=key)
    sensory_stats = SensoryStatistics(
        audio_tokens=cfg.n_audio_tokens,
        vision_tokens=cfg.n_vision_tokens,
        codebook_size=cfg.codebook_size_audio)
    aligner = ContrastiveAligner(embed_dim=cfg.d_model, buffer_size=16, tau=0.07)
    predictor = PredictiveProcessor(lr=1e-5)

    # Fill buffer
    for i in range(4):
        aligner.push_text_emb(jax.random.normal(jax.random.PRNGKey(10 + i), (cfg.d_model,)))

    text_tokens = jax.random.normal(key, (cfg.n_tokens, cfg.d_model))
    audio_raw = jax.random.normal(key, (32000,))  # simulated TTS audio
    vision_raw = jax.random.normal(key, (224, 224, 3))

    tokens, sense_info = sense_module.process_and_inject(text_tokens, audio_raw, vision_raw)
    sensory_stats.update(sense_info["audio_indices"], sense_info["vision_indices"])

    key, sk = jax.random.split(key)
    carry, (h_out, obs, q_final, q_data) = halo3_step(model, carry, tokens, sk)

    key, lk = jax.random.split(key)
    model, sense_module, loss, learn_info = predictor.learn_from_error(
        model, sense_module, carry, text_tokens, audio_raw, vision_raw, q_data, lk,
        contrastive_aligner=aligner, text_paired=True, contrastive_weight=0.3)

    assert np.isfinite(loss)
    assert not aligner.matured  # not yet
```

- [ ] **Step 2: Run full test suite**

Run: `cd /d/New_Ai/.worktrees/halo3 && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
cd /d/New_Ai/.worktrees/halo3
git add tests/senses/test_integration.py
git commit -m "test(senses): Phase B integration test — TTS + contrastive alignment"
```

---

### Task 10: Delete Old v3.7 Checkpoint and Verify

**Files:** None (operational task)

- [ ] **Step 1: Delete old sense_module checkpoint**

The v3.7 checkpoint has 32-code audio codebook and 8 tokens — incompatible with v3.8. Delete it so a fresh checkpoint is created on first run.

```bash
rm -f D:/New_Ai/.worktrees/halo3/data/checkpoints/sense_module.eqx
```

- [ ] **Step 2: Build Docker image**

```bash
cd /d/New_Ai/.worktrees/halo3
docker rm -f halo3-train-1
MSYS_NO_PATHCONV=1 docker compose build train
```

- [ ] **Step 3: Start and verify**

```bash
MSYS_NO_PATHCONV=1 docker compose up -d train
docker logs -f halo3-train-1
```

Watch for:
- `Senses: FNO spectral cortex (critical period)` — fresh init
- `TTS: espeak ON` — TTS available
- `[T]` in sense labels — TTS narration active
- Codebook utilization climbing over ticks

- [ ] **Step 4: Monitor VRAM**

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

Expected: ~5340 MiB (trivial increase from v3.7's 5338 MiB)
