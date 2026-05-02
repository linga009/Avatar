import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType


class JambaBackbone(nn.Module):
    """
    Jamba hybrid Transformer+Mamba backbone loaded in 4-bit NF4 with QLoRA.
    LoRA (r=16, alpha=32) applied to attention q/k/v/o projections.
    Returns last-layer hidden states for downstream heads.
    """

    def __init__(self, model_id: str = "ai21labs/Jamba-v0.1",
                 lora_r: int = 16, lora_alpha: int = 32):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base, lora_cfg)

    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (B, seq_len) token IDs
        Returns: (B, seq_len, 4096) last-layer hidden states
        """
        out = self.model(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states[-1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_hidden_states(input_ids)
