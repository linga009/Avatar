# halo/config.py
from dataclasses import dataclass


@dataclass
class HaloConfig:
    # Dimensions
    d_model: int = 256
    d_boundary: int = 64
    d_head: int = 64
    n_heads: int = 4

    # Backbone
    n_layers: int = 8        # Pattern: [S,S,S,H,S,S,S,H]
    d_state: int = 16        # SSM state dimension
    d_ff: int = 512          # Feed-forward hidden dimension

    # Modality conformal dimensions (initial values; learned during training)
    delta_text: float = 1.0
    delta_image: float = 2.0
    delta_flow: float = 1.5  # KG prior conformal dimension

    # Flow matching
    flow_steps: int = 4      # Euler ODE steps at inference

    # Memory
    max_cache: int = 128
    island_size: int = 32
    bekenstein_alpha: float = 0.1

    # Encoders
    clip_model: str = "ViT-L/14"
    clip_pretrained: str = "openai"
    vocab_size: int = 50257
    text_embed_dim: int = 768
    image_embed_dim: int = 768

    # Training
    lr: float = 3e-4
    fisher_lambda: float = 1e-3
    lambda_bek: float = 0.1
    lambda_thermo: float = 0.05
    lambda_page: float = 0.05
