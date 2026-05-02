import torch
import pytest
from unittest.mock import patch, MagicMock
from med_rai.backbone.jamba_backbone import JambaBackbone

D_JAMBA = 4096

def make_mock_model():
    """Create a minimal mock that mimics Jamba's interface."""
    mock = MagicMock()
    # Simulate output_hidden_states=True response
    mock_out = MagicMock()
    mock_out.hidden_states = [torch.randn(1, 8, D_JAMBA)] * 33  # 33 layers
    mock.return_value = mock_out
    return mock

@patch("med_rai.backbone.jamba_backbone.AutoModelForCausalLM.from_pretrained")
@patch("med_rai.backbone.jamba_backbone.get_peft_model")
def test_hidden_states_shape(mock_peft, mock_pretrained):
    mock_model = make_mock_model()
    mock_pretrained.return_value = MagicMock()
    mock_peft.return_value = mock_model

    backbone = JambaBackbone(model_id="ai21labs/Jamba-v0.1")
    token_ids = torch.randint(0, 100, (1, 8))
    h = backbone.get_hidden_states(token_ids)
    assert h.shape == (1, 8, D_JAMBA), f"Expected (1,8,{D_JAMBA}), got {h.shape}"

@patch("med_rai.backbone.jamba_backbone.AutoModelForCausalLM.from_pretrained")
@patch("med_rai.backbone.jamba_backbone.get_peft_model")
def test_forward_equals_get_hidden_states(mock_peft, mock_pretrained):
    mock_model = make_mock_model()
    mock_pretrained.return_value = MagicMock()
    mock_peft.return_value = mock_model

    backbone = JambaBackbone(model_id="ai21labs/Jamba-v0.1")
    token_ids = torch.randint(0, 100, (1, 8))
    h1 = backbone.get_hidden_states(token_ids)
    h2 = backbone(token_ids)
    assert torch.allclose(h1, h2)

@patch("med_rai.backbone.jamba_backbone.AutoModelForCausalLM.from_pretrained")
@patch("med_rai.backbone.jamba_backbone.get_peft_model")
def test_bnb_config_used(mock_peft, mock_pretrained):
    """Verify BitsAndBytesConfig is passed to from_pretrained."""
    mock_pretrained.return_value = MagicMock()
    mock_peft.return_value = MagicMock()

    JambaBackbone(model_id="ai21labs/Jamba-v0.1")
    call_kwargs = mock_pretrained.call_args[1]
    assert "quantization_config" in call_kwargs, "BitsAndBytesConfig must be passed"
