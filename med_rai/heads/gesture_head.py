import torch
import torch.nn as nn

GESTURE_NAMES = [
    "reaching_for_needle_with_right_hand",
    "positioning_needle",
    "pushing_needle_through_tissue",
    "transferring_needle_from_left_to_right",
    "moving_to_center",
    "pulling_suture_with_left_hand",
    "pulling_suture_with_right_hand",
    "orienting_needle",
    "using_right_hand_to_help_tighten_suture",
    "loosening_more_suture",
    "dropping_suture_at_end_and_moving_to_center",
    "reaching_for_needle_with_left_hand",
    "making_figure_of_eight",
    "pulling_needle_out_of_tissue",
    "idle",
]


class GestureHead(nn.Module):
    """Linear classifier over JIGSAWS 15-gesture vocabulary."""

    def __init__(self, d_jamba: int = 4096, n_gestures: int = 15):
        super().__init__()
        self.classifier = nn.Linear(d_jamba, n_gestures)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, S, d_jamba) -> logits: (B, n_gestures) via mean pooling"""
        pooled = h.mean(dim=1)           # (B, d_jamba)
        return self.classifier(pooled)   # (B, n_gestures)
