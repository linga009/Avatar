# halo/backbone/halo_backbone.py
import torch
import torch.nn as nn
from halo.config import HaloConfig
from halo.backbone.simple_ssm import SimpleSSM
from halo.attention.holo_attention import HoloAttention


class HALOBackbone(nn.Module):
    """Hybrid backbone alternating SimpleSSM and HoloAttention.

    Layer pattern for n_layers=8: [S, S, S, H, S, S, S, H]
    Each layer: LayerNorm -> core -> residual -> LayerNorm -> FFN -> residual.
    """

    def __init__(self, cfg: HaloConfig) -> None:
        super().__init__()
        # Build layer descriptors: type + module
        self.layers: list[dict] = []
        core_modules = nn.ModuleList()
        norms1 = nn.ModuleList()
        norms2 = nn.ModuleList()
        ffns = nn.ModuleList()

        period = 4  # HoloAttention at every 4th position (indices 3, 7, ...)
        for i in range(cfg.n_layers):
            if (i + 1) % period == 0:
                core = HoloAttention(cfg)
                layer_type = "holo"
            else:
                core = SimpleSSM(cfg)
                layer_type = "ssm"
            self.layers.append({"type": layer_type, "idx": i})
            core_modules.append(core)
            norms1.append(nn.LayerNorm(cfg.d_model))
            norms2.append(nn.LayerNorm(cfg.d_model))
            ffns.append(nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model),
            ))

        self.core_modules = core_modules
        self.norms1 = norms1
        self.norms2 = norms2
        self.ffns = ffns

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N, d_model) token embeddings
            x: (B, N, d_boundary) boundary coordinates
            z: (B, N, 1) depths
        Returns:
            h: (B, N, d_model) transformed embeddings
        """
        for i, layer_info in enumerate(self.layers):
            # Pre-norm
            h_norm = self.norms1[i](h)

            # Core layer
            if layer_info["type"] == "holo":
                core_out, _ = self.core_modules[i](h_norm, x, z)
            else:
                core_out = self.core_modules[i](h_norm)

            h = h + core_out  # residual

            # FFN
            h = h + self.ffns[i](self.norms2[i](h))  # residual

        return h
