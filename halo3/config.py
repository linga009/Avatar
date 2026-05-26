from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Halo3Config:
    """Immutable configuration for HoloBiont 3.0 Physics Engine."""

    # Backbone — maximized for 6 GB VRAM (no JIT recompile needed)
    d_model: int = 2048
    d_boundary: int = 64
    n_heads: int = 16
    d_head: int = 128
    n_layers: int = 60
    d_state: int = 256
    layer_pattern: str = "SSSSSH"
    n_shared_attn: int = 2
    lora_rank: int = 16
    reversible: bool = True

    # MERA-FFN
    mera_bond_dim: int = 64
    mera_n_cores: int = 4

    # Lorentz
    init_curvature: float = 1.0

    # Hamiltonian
    n_leapfrog_steps: int = 3
    leapfrog_step_size: float = 0.1
    lambda_energy: float = 0.1

    # Kuramoto
    n_clusters: int = 32
    n_hidden: int = 16
    kuramoto_dt: float = 0.1
    init_coupling: float = 0.3
    lambda_sync: float = 0.01

    # Page memory
    max_cache: int = 128
    island_size: int = 32

    # Training
    lr: float = 3e-4
    n_steps: int = 10_000
    seed: int = 42
    galore_rank: int = 64
    lisa_active_layers: int = 2

    # Language model
    vocab_size: int = 8000
    max_seq_len: int = 128

    # Bridges
    n_tokens: int = 32
    n_obs: int = 8
    n_actions: int = 8

    # FNO (Fourier Neural Operator) — sensory perception
    fno_hidden_dim: int = 64
    fno_n_layers: int = 4
    fno_audio_modes: int = 32       # v3.8: was 16
    fno_vision_modes: int = 16      # v3.9: was 8 (doubled for richer spectral vision)

    # VQ-VAE — spectral codebook (split per modality for v3.8)
    codebook_size_audio: int = 128  # v3.8: was 32 (shared)
    codebook_size_vision: int = 64  # v3.9: was 32 (doubled to match richer vision tokens)
    codebook_dim: int = 64
    codebook_ema_decay: float = 0.99
    commitment_beta: float = 0.25
    dead_code_threshold: int = 100

    # Sense tokens
    n_audio_tokens: int = 16        # v3.8: was 8
    n_vision_tokens: int = 8        # v3.9: was 4 (doubled for richer spectral vision)

    # Critical period
    critical_period_recon_weight: float = 0.5

    # TTS self-narration (v3.8)
    tts_mode: str = "kokoro"        # "kokoro" (Phase C, neural 82M) or "espeak" (Phase B, rule-based)
    tts_every_n: int = 3            # use TTS every Nth tick when mic active

    # Contrastive alignment (v3.8)
    contrastive_tau: float = 0.07
    contrastive_weight: float = 0.3
    contrastive_maturation_threshold: float = 0.75

    # Meta-layer
    meta_n_hidden: int = 8
    meta_n_actions: int = 4
    meta_k: int = 10

    # Homeostatic
    homeo_sync_threshold: float = 0.6
    homeo_blend_clip: float = 1.0

    # COP (Critical Order-Parameter Cognition) — v4.0
    cop_window: int = 20
    cop_eta: float = 0.0005
    cop_K_min: float = 0.05
    cop_K_max: float = 2.0
    cop_coherence_ema: float = 0.1
    cop_warmup: int = 5

    # Heartbeat
    tick_interval: int = 60

    def __post_init__(self) -> None:
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(
                f"n_heads ({self.n_heads}) * d_head ({self.d_head}) "
                f"must equal d_model ({self.d_model})"
            )
        if self.n_layers % len(self.layer_pattern) != 0:
            raise ValueError(
                f"n_layers ({self.n_layers}) must be divisible by "
                f"len(layer_pattern) ({len(self.layer_pattern)})"
            )
        if self.n_leapfrog_steps < 1:
            raise ValueError(f"n_leapfrog_steps must be >= 1")
        if self.meta_k < 1:
            raise ValueError(f"meta_k must be >= 1")
