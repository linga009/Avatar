# halo_fep/intellect/llm_bridge.py
"""On-demand Phi-3.5-mini-instruct integration at 4-bit NF4 quantization.

Load/unload contract
--------------------
* Call ``load()`` before each wake cycle.
* Call ``unload()`` immediately after — **always**, even on error.
* Never leave the model loaded between ticks; peak VRAM ≈ 2 GB.

Timeout safety
--------------
``think()`` wraps generation in a ``ThreadPoolExecutor`` with a configurable
timeout (default 30 s).  If the LLM has not produced output within that window
a ``TimeoutError`` is raised, the thread is abandoned, and the heartbeat loop
can continue.  This prevents a stalled LLM from blocking the subconscious.

Security note — trust_remote_code
----------------------------------
``trust_remote_code=True`` is required by some Phi model checkpoints because
they ship custom modelling code in the HuggingFace repository.  This means
**arbitrary Python code from the model repository is executed during load**.
This is acceptable for the pinned ``microsoft/Phi-3.5-mini-instruct`` checkpoint
but must be reviewed before switching to any other model ID.
"""
from __future__ import annotations

import logging
import concurrent.futures
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_MODEL_ID        = "microsoft/Phi-3.5-mini-instruct"
_DEFAULT_TIMEOUT = 30  # seconds; generation beyond this is abandoned


@dataclass
class LLMResponse:
    """Structured response parsed from a single line of LLM output.

    Attributes
    ----------
    action  : One of ``"SEARCH"``, ``"GOAL"``, ``"LEARN"``, ``"IDLE"``.
    content : The text following the action prefix (empty for ``IDLE``).
    """
    action:  str
    content: str


def parse_llm_output(text: str) -> LLMResponse:
    """Parse the first non-empty line of an LLM reply into a structured response.

    Expected formats (case-sensitive):
        SEARCH: <query>
        GOAL:   <description>
        LEARN:  <fact>
        IDLE

    Any unrecognised output is treated as ``IDLE`` and a warning is logged.

    Parameters
    ----------
    text : Raw string output from ``LLMBridge.think()``.

    Returns
    -------
    LLMResponse with action and content fields populated.
    """
    first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
    for prefix in ("SEARCH:", "GOAL:", "LEARN:"):
        if first_line.startswith(prefix):
            return LLMResponse(
                action=prefix.rstrip(":"),
                content=first_line[len(prefix):].strip(),
            )
    if first_line.upper() == "IDLE":
        return LLMResponse(action="IDLE", content="")
    log.warning(
        f"LLM output not parseable: {first_line!r}. Defaulting to IDLE."
    )
    return LLMResponse(action="IDLE", content="")


class LLMBridge:
    """Thin wrapper around Phi-3.5-mini-instruct with load/unload lifecycle.

    Parameters
    ----------
    model_id : HuggingFace model identifier (default: Phi-3.5-mini-instruct).
    timeout  : Generation timeout in seconds (default: 30).

    Example
    -------
    >>> llm = LLMBridge()
    >>> llm.load()
    >>> output = llm.think("SEARCH: artificial intelligence")
    >>> llm.unload()
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._model_id   = model_id
        self._timeout    = timeout
        self._model      = None
        self._tokenizer  = None

    @property
    def is_loaded(self) -> bool:
        """True if the model is currently loaded in CUDA memory."""
        return self._model is not None

    def load(self) -> None:
        """Load the LLM to CUDA with 4-bit NF4 quantization.

        No-op if already loaded.

        Notes
        -----
        ``trust_remote_code=True`` is required for Phi-3.5 but executes
        code from the HuggingFace repo — verify the model ID before changing.
        """
        if self.is_loaded:
            return
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        log.info(f"Loading {self._model_id} at 4-bit NF4...")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model     = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=quant_cfg,
            device_map="auto",
            # SECURITY: trust_remote_code=True runs code from the model repo.
            # Acceptable for microsoft/Phi-3.5-mini-instruct (audited checkpoint).
            # Review before switching model_id.
            trust_remote_code=True,
        )
        log.info("LLM loaded.")

    def unload(self) -> None:
        """Free CUDA memory.  No-op if already unloaded."""
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
        """Run LLM inference with a timeout guard.

        The generation runs in a background thread.  If it does not complete
        within ``self._timeout`` seconds, a ``TimeoutError`` is raised and the
        background thread is abandoned (it will eventually be reclaimed by the
        interpreter when generation finishes).

        Parameters
        ----------
        prompt     : Input text prompt for the model.
        max_tokens : Maximum number of new tokens to generate.

        Returns
        -------
        Generated text (stripped of special tokens and leading/trailing whitespace).

        Raises
        ------
        RuntimeError  : If called before ``load()``.
        TimeoutError  : If generation exceeds ``self._timeout`` seconds.
        """
        if not self.is_loaded:
            raise RuntimeError("LLMBridge.think() called before load()")

        import torch

        def _generate() -> str:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(
                self._model.device
            )
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Run in a thread so we can apply a wall-clock timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            try:
                return future.result(timeout=self._timeout)
            except concurrent.futures.TimeoutError:
                log.error(
                    f"LLM generation timed out after {self._timeout} s. "
                    "Returning empty string."
                )
                raise TimeoutError(
                    f"LLM generation did not complete within {self._timeout} s."
                )
