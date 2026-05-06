"""On-demand Phi-3.5-mini-instruct integration at 4-bit NF4 quantization.

Load/unload contract:
  - load() before wake cycle, unload() after (always, even on error).
  - Peak VRAM: ~2GB. Never leave model loaded between ticks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"


@dataclass
class LLMResponse:
    action:  str   # "SEARCH" | "GOAL" | "LEARN" | "IDLE"
    content: str   # the text after the prefix


def parse_llm_output(text: str) -> LLMResponse:
    """Parse first line of LLM reply into structured LLMResponse."""
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    for prefix in ("SEARCH:", "GOAL:", "LEARN:"):
        if first_line.startswith(prefix):
            return LLMResponse(
                action=prefix.rstrip(":"),
                content=first_line[len(prefix):].strip(),
            )
    if first_line.upper() == "IDLE":
        return LLMResponse(action="IDLE", content="")
    log.warning(f"LLM output not parseable: {first_line!r}. Defaulting to IDLE.")
    return LLMResponse(action="IDLE", content="")


class LLMBridge:
    """Thin wrapper around Phi-3.5-mini-instruct with load/unload lifecycle."""

    def __init__(self) -> None:
        self._model     = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load Phi-3.5-mini to CUDA with 4-bit NF4 quantization."""
        if self.is_loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        log.info(f"Loading {_MODEL_ID} at 4-bit NF4...")
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
        self._model     = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        log.info("LLM loaded.")

    def unload(self) -> None:
        """Free CUDA memory."""
        if not self.is_loaded:
            return
        import torch
        del self._model
        del self._tokenizer
        self._model     = None
        self._tokenizer = None
        torch.cuda.empty_cache()
        log.info("LLM unloaded.")

    def think(self, prompt: str, max_tokens: int = 128) -> str:
        """Run inference. Returns raw text output."""
        if not self.is_loaded:
            raise RuntimeError("LLMBridge.think() called before load()")
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
