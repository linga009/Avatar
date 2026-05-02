import torch
import torch.nn as nn
from typing import Tuple, Dict


class MedRAILoss(nn.Module):
    """
    Weighted multi-task loss:
        L_total = lambda_rfm * L_rfm + lambda_text * L_text + lambda_gesture * L_gesture

    L_rfm:     Flow Matching MSE in se(3): ||v_theta - (xi_1 - xi_0)||^2
    L_text:    Cross-entropy over vocabulary
    L_gesture: Cross-entropy over 15 JIGSAWS gestures
    """

    def __init__(self, lambda_rfm: float = 1.0,
                 lambda_text: float = 0.5,
                 lambda_gesture: float = 0.3):
        super().__init__()
        self.lambda_rfm = lambda_rfm
        self.lambda_text = lambda_text
        self.lambda_gesture = lambda_gesture
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        v_pred: torch.Tensor,
        u_target: torch.Tensor,
        text_logits: torch.Tensor,
        text_targets: torch.Tensor,
        gesture_logits: torch.Tensor,
        gesture_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        l_rfm = ((v_pred - u_target) ** 2).mean()
        l_text = self.ce(text_logits.reshape(-1, text_logits.size(-1)),
                         text_targets.reshape(-1))
        l_gesture = self.ce(gesture_logits, gesture_targets)

        total = (self.lambda_rfm * l_rfm
                 + self.lambda_text * l_text
                 + self.lambda_gesture * l_gesture)
        return total, {"rfm": l_rfm, "text": l_text, "gesture": l_gesture}
