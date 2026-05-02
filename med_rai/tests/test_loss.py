import torch
import pytest
from med_rai.loss import MedRAILoss

B, H, SE3_DIM, SEQ, VOCAB, N_GESTURES = 2, 10, 6, 19, 65536, 15

@pytest.fixture
def loss_fn():
    return MedRAILoss(lambda_rfm=1.0, lambda_text=0.5, lambda_gesture=0.3)

def test_loss_shape(loss_fn):
    v_pred = torch.randn(B, H, SE3_DIM)
    u_target = torch.randn(B, H, SE3_DIM)
    text_logits = torch.randn(B, SEQ, VOCAB)
    text_targets = torch.randint(0, VOCAB, (B, SEQ))
    gesture_logits = torch.randn(B, N_GESTURES)
    gesture_targets = torch.randint(0, N_GESTURES, (B,))
    total, components = loss_fn(v_pred, u_target, text_logits,
                                text_targets, gesture_logits, gesture_targets)
    assert total.shape == (), "Total loss must be scalar"
    assert set(components.keys()) == {"rfm", "text", "gesture"}

def test_loss_weights(loss_fn):
    v_pred = torch.zeros(1, H, SE3_DIM)
    u_target = torch.ones(1, H, SE3_DIM)
    text_logits = torch.zeros(1, SEQ, VOCAB)
    text_targets = torch.zeros(1, SEQ, dtype=torch.long)
    gesture_logits = torch.zeros(1, N_GESTURES)
    gesture_targets = torch.zeros(1, dtype=torch.long)
    total, c = loss_fn(v_pred, u_target, text_logits,
                       text_targets, gesture_logits, gesture_targets)
    expected = 1.0 * c["rfm"] + 0.5 * c["text"] + 0.3 * c["gesture"]
    assert torch.isclose(total, expected, atol=1e-5)

def test_no_nan_loss(loss_fn):
    v_pred = torch.randn(B, H, SE3_DIM)
    u_target = torch.randn(B, H, SE3_DIM)
    text_logits = torch.randn(B, SEQ, VOCAB)
    text_targets = torch.randint(0, VOCAB, (B, SEQ))
    gesture_logits = torch.randn(B, N_GESTURES)
    gesture_targets = torch.randint(0, N_GESTURES, (B,))
    total, _ = loss_fn(v_pred, u_target, text_logits,
                       text_targets, gesture_logits, gesture_targets)
    assert not torch.isnan(total)
