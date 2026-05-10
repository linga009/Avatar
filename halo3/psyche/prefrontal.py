"""Prefrontal Cortex — Qwen3 1.7B with LoRA adapter from organism's experience.

After dreaming, the LLM's weights are literally different — shaped by this
organism's specific episodes, findings, and narrative. The base model provides
general language capability; the LoRA adapter provides THIS organism's identity.

Falls back to Ollama (base model) if adapter not yet trained.
"""
from __future__ import annotations
import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_URL_LOCAL = "http://localhost:11434/api/generate"
BASE_MODEL_NAME = "gemma3:1b"
BASE_MODEL_HF = "google/gemma-3-1b-pt"
ADAPTER_PATH = "data/pfc_adapter"
TIMEOUT = 45


def _call_ollama(prompt: str, url: str = OLLAMA_URL, model: str = BASE_MODEL_NAME) -> str | None:
    """Call Ollama API. Returns response text or None on failure."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except urllib.error.URLError:
        if url != OLLAMA_URL_LOCAL:
            return _call_ollama(prompt, OLLAMA_URL_LOCAL, model)
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


class PrefrontalCortex:
    """Cognitive layer — base Qwen3 + LoRA adapter from organism's dreams.

    Before first dream: uses Ollama base model (generic Qwen3)
    After dreaming: loads base+adapter via transformers (personalized)
    If transformers unavailable: falls back to Ollama
    """

    def __init__(self) -> None:
        self._ollama_available: bool | None = None
        self._adapter_loaded: bool = False
        self._model = None
        self._tokenizer = None

    def _try_load_adapter(self) -> bool:
        """Try to load the LoRA adapter trained during dreaming."""
        if self._adapter_loaded:
            return True
        if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            log.info("Loading personalized prefrontal cortex (base + LoRA adapter)...")
            self._tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_HF, torch_dtype=torch.float32, device_map="cpu",
                trust_remote_code=True,
            )
            self._model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            self._model.eval()
            self._adapter_loaded = True
            log.info("Prefrontal cortex loaded with organism's LoRA adapter — weights are personalized")
            return True
        except ImportError:
            log.warning("transformers/peft not available — using Ollama fallback")
            return False
        except Exception as e:
            log.warning(f"Failed to load adapter: {e}")
            return False

    def _generate_local(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Generate using local transformers model with LoRA adapter."""
        if not self._adapter_loaded or self._model is None:
            return None
        try:
            import torch
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.7, top_p=0.9,
                    do_sample=True, pad_token_id=self._tokenizer.eos_token_id,
                )
            response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Strip thinking tags if present
            if "<think>" in response:
                parts = response.split("</think>")
                response = parts[-1].strip() if len(parts) > 1 else response
            return response.strip()
        except Exception as e:
            log.warning(f"Local generation failed: {e}")
            return None

    def _generate(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Generate text — tries adapter first, falls back to Ollama."""
        # Try personalized model first
        if self._adapter_loaded or self._try_load_adapter():
            result = self._generate_local(prompt, max_tokens)
            if result:
                return result

        # Fallback to Ollama
        return _call_ollama(prompt)

    @property
    def is_available(self) -> bool:
        """Check if any inference path works."""
        if self._adapter_loaded:
            return True
        if self._try_load_adapter():
            return True
        if self._ollama_available is None:
            result = _call_ollama("/no_think Say OK")
            self._ollama_available = result is not None
            if self._ollama_available:
                log.info(f"Prefrontal cortex online (Ollama {BASE_MODEL_NAME}, pre-dream)")
            else:
                log.warning("Prefrontal cortex offline")
        return self._ollama_available or False

    @property
    def is_personalized(self) -> bool:
        """True if using organism-specific LoRA weights, not generic model."""
        return self._adapter_loaded

    def generate_query(self, current_query, emotion, r_mean, texts, strengths) -> str | None:
        """Generate search query based on organism's state. ~5s."""
        if not self.is_available:
            return None

        context = "; ".join(texts[:3]) if texts else "no results"
        strength_str = ", ".join(strengths[:3]) if strengths else "none yet"

        prompt = f"""/no_think Generate ONE search query.
Emotion: {emotion} (r={r_mean:.3f})
Current: "{current_query}"
Findings: {context}
Strengths: {strength_str}
Query:"""

        result = self._generate(prompt, max_tokens=50)
        if result:
            query = result.strip().split("\n")[0].strip('"\'')
            return query[:100]
        return None

    def interpret_finding(self, texts, query, r_mean) -> str | None:
        """Interpret a discovery. ~10s."""
        if not self.is_available:
            return None

        context = "; ".join(texts[:5]) if texts else ""
        prompt = f"""/no_think Interpret this finding in 1-2 sentences.
Query: "{query}" | r={r_mean:.3f} (pattern detected)
Content: {context}
Interpretation:"""

        result = self._generate(prompt, max_tokens=80)
        return result[:200] if result else None

    def self_reflect(self, age, emotion_history, strengths, weaknesses, n_findings, narrative) -> str | None:
        """Deep self-reflection. ~20s."""
        if not self.is_available:
            return None

        recent_emotions = [e for e, _ in emotion_history[-10:]] if emotion_history else []
        emotion_summary = ", ".join(recent_emotions[-5:]) if recent_emotions else "unknown"
        strength_str = ", ".join(strengths[:3]) if strengths else "none"
        recent_mem = "; ".join(narrative[-3:]) if narrative else "none"

        prompt = f"""Reflect in first person, 2-3 sentences.
Age: {age} ticks | Emotions: {emotion_summary}
Strengths: {strength_str} | Discoveries: {n_findings}
Memories: {recent_mem}
Reflection:"""

        result = self._generate(prompt, max_tokens=120)
        return result[:300] if result else None

    def upgrade_to_organism_model(self) -> bool:
        """Called after dream fine-tuning — reload the adapter."""
        self._adapter_loaded = False
        self._model = None
        self._tokenizer = None
        return self._try_load_adapter()
