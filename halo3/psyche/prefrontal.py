"""Prefrontal Cortex — Qwen3 0.6B with LoRA adapter from organism's experience.

After dreaming, the LLM's weights are literally different — shaped by this
organism's specific episodes, findings, and narrative. The base model provides
general language capability; the LoRA adapter provides THIS organism's identity.

Falls back to Ollama (base model) if adapter not yet trained or quality is low.

v3.1 fixes:
  - Format-matched generation: local model uses ### Instruction / ### Response
  - Quality gating: garbage output falls through to Ollama
  - Semantic dedup: rejects queries too similar to recent ones
  - Failure context: tells PFC when searches are returning nothing
"""
from __future__ import annotations
import json
import logging
import os
import urllib.request
import urllib.error
from collections import deque

log = logging.getLogger(__name__)

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_URL_LOCAL = "http://localhost:11434/api/generate"
BASE_MODEL_NAME = "qwen3:0.6b"
BASE_MODEL_HF = "Qwen/Qwen3-0.6B"
ADAPTER_PATH = "data/pfc_adapter"
TIMEOUT = 45

# Words that should never appear in a search query
_JUNK_WORDS = frozenset([
    "**query:**", "query:", "**", "```", "/no_think", "/think",
    "search query:", "here is", "here's", "### response:",
    "### instruction:", "response:",
])
_STATE_WORDS = frozenset([
    "pride", "anxiety", "boredom", "satisfaction", "curiosity",
    "synchronization", "feeling", "organism", "emotion",
])


def _call_ollama(prompt: str, url: str = OLLAMA_URL, model: str = BASE_MODEL_NAME) -> str | None:
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


def _clean_query(raw: str) -> str | None:
    """Clean a raw LLM output into a usable search query. Returns None if unrecoverable."""
    if not raw:
        return None
    query = raw.strip().split("\n")[0]
    # Strip formatting and meta-text
    low = query.lower()
    for junk in _JUNK_WORDS:
        low_junk = junk.lower()
        if low_junk in low:
            idx = low.find(low_junk)
            query = query[:idx] + query[idx + len(junk):]
            low = query.lower()
    # Strip state words from start
    for sw in _STATE_WORDS:
        if low.startswith(sw):
            query = query[len(sw):]
            low = query.lower().strip()
    query = query.strip('"\'*:/ \t')
    # Strip URLs
    if query.startswith("http") or "google.com" in query or "search?q=" in query:
        if "q=" in query:
            try:
                import urllib.parse
                qs = query.split("q=", 1)[1].split("&")[0]
                query = urllib.parse.unquote_plus(qs)[:80]
            except Exception:
                return None
        else:
            return None
    if len(query) < 5 or query.startswith("/") or query.startswith("*"):
        return None
    return query[:80]


def _query_quality(query: str) -> float:
    """Score query quality 0-1. Low scores = likely garbage."""
    score = 1.0
    words = query.split()
    if len(words) < 2:
        score -= 0.4
    if len(words) > 15:
        score -= 0.3
    # Penalize if contains state/meta words
    low = query.lower()
    for sw in _STATE_WORDS:
        if sw in low:
            score -= 0.3
    # Penalize if too many special characters
    special = sum(1 for c in query if c in '"\'()[]{}|\\<>')
    if special > 3:
        score -= 0.3
    # Penalize if contains prompt artifacts
    if "instruction" in low or "response" in low or "topic:" in low:
        score -= 0.5
    return max(0.0, score)


def _cosine_sim_simple(a: str, b: str) -> float:
    """Simple word-overlap cosine similarity between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    overlap = len(wa & wb)
    return overlap / (len(wa) ** 0.5 * len(wb) ** 0.5)


class PrefrontalCortex:
    """Cognitive layer — base Qwen3 + LoRA adapter from organism's dreams.

    Before first dream: uses Ollama base model (generic Qwen3)
    After dreaming: loads base+adapter via transformers (personalized)
    Quality gating ensures garbage adapter output falls through to Ollama.
    """

    def __init__(self) -> None:
        self._ollama_available: bool | None = None
        self._adapter_loaded: bool = False
        self._model = None
        self._tokenizer = None
        self._instructions: dict | None = None
        self._recent_queries: deque = deque(maxlen=10)
        self._query_successes: int = 0
        self._query_attempts: int = 0

    @property
    def query_success_rate(self) -> float:
        if self._query_attempts == 0:
            return 1.0
        return self._query_successes / self._query_attempts

    def record_query_result(self, had_results: bool) -> None:
        """Called by organism to track whether PFC queries actually work."""
        self._query_attempts += 1
        if had_results:
            self._query_successes += 1

    def _get_instructions(self) -> dict:
        if self._instructions is None:
            try:
                from halo3.training.dream_gepa import load_prompt_instructions
                self._instructions = load_prompt_instructions()
            except Exception:
                self._instructions = {
                    "query_instruction": (
                        "Output ONLY a web search query of 5-8 words. "
                        "No labels, no explanation, just search terms."
                    ),
                    "reflection_instruction": (
                        "Reflect in first person, 2-3 sentences."
                    ),
                }
        return self._instructions

    def reload_instructions(self) -> None:
        self._instructions = None

    def _try_load_adapter(self) -> bool:
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
            log.info("Prefrontal cortex loaded with organism's LoRA adapter")
            return True
        except ImportError:
            log.warning("transformers/peft not available — using Ollama fallback")
            return False
        except Exception as e:
            log.warning(f"Failed to load adapter: {e}")
            return False

    def _generate_local(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Generate using local transformers model with LoRA adapter.

        CRITICAL FIX: Uses ### Instruction / ### Response format to match
        the LoRA training data format.
        """
        if not self._adapter_loaded or self._model is None:
            return None
        try:
            import torch
            # Wrap prompt in training format so LoRA weights are relevant
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = self._tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.7, top_p=0.9,
                    do_sample=True, pad_token_id=self._tokenizer.eos_token_id,
                )
            response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if "<think>" in response:
                parts = response.split("</think>")
                response = parts[-1].strip() if len(parts) > 1 else response
            return response.strip()
        except Exception as e:
            log.warning(f"Local generation failed: {e}")
            return None

    def _generate(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Generate text — tries adapter first with quality gating, falls back to Ollama.

        CRITICAL FIX: If adapter output fails quality check, falls through to Ollama
        instead of accepting garbage.
        """
        # Try personalized model first
        if self._adapter_loaded or self._try_load_adapter():
            result = self._generate_local(prompt, max_tokens)
            if result:
                quality = _query_quality(result)
                if quality >= 0.5:
                    return result
                log.debug(f"PFC adapter output low quality ({quality:.2f}): '{result[:40]}' — trying Ollama")

        # Fallback to Ollama
        return _call_ollama(f"/no_think {prompt}")

    @property
    def is_available(self) -> bool:
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
        return self._adapter_loaded

    def generate_query(
        self,
        current_query: str,
        emotion: str,
        r_mean: float,
        texts: list[str],
        strengths: list[str],
        consecutive_failures: int = 0,
        dead_queries: list[str] | None = None,
    ) -> str | None:
        """Generate search query based on organism's state.

        Args:
            consecutive_failures: how many ticks in a row had zero search results
            dead_queries: queries known to return nothing (from negative memory)
        """
        if not self.is_available:
            return None

        # Build context
        snippets = []
        for t in texts[:3]:
            t = t.strip()
            if len(t) > 30:
                snippets.append(t[:80])
            if len(snippets) >= 2:
                break
        context = snippets[0] if snippets else ""

        strength_str = ", ".join(strengths[:3]) if strengths else "general research"

        # Build failure context if searches are returning nothing
        failure_warning = ""
        if consecutive_failures >= 3:
            failure_warning = (
                f"\nWARNING: The last {consecutive_failures} searches returned ZERO results. "
                f"The query '{current_query}' is a dead end. "
                f"You MUST choose a completely different topic."
            )

        dead_warning = ""
        if dead_queries:
            dead_warning = f"\nAvoid these dead-end topics: {', '.join(dead_queries[:5])}"

        instr = self._get_instructions().get(
            "query_instruction",
            "Output ONLY a web search query of 5-8 words. No labels, no explanation.",
        )

        # Build prompt for PFC
        prompt_parts = [
            f"{instr}",
            f"\nState: feeling {emotion}, synchronization {r_mean:.2f}",
            f"Current topic: {current_query}",
        ]
        if context:
            prompt_parts.append(f"Recent findings: {context}")
        prompt_parts.append(f"Interests: {strength_str}")
        if failure_warning:
            prompt_parts.append(failure_warning)
        if dead_warning:
            prompt_parts.append(dead_warning)
        prompt_parts.append("\nSearch query:")

        prompt = "\n".join(prompt_parts)

        # Generate candidate(s)
        result = self._generate(prompt, max_tokens=20)
        if not result:
            return None

        query = _clean_query(result)
        if not query:
            return None

        # Quality gate
        if _query_quality(query) < 0.5:
            log.debug(f"PFC query rejected (low quality): '{query[:40]}'")
            return None

        # Semantic dedup: reject if too similar to recent queries
        for recent in self._recent_queries:
            if _cosine_sim_simple(query, recent) > 0.8:
                log.debug(f"PFC query rejected (too similar to recent): '{query[:40]}'")
                return None

        # Check against known dead queries
        if dead_queries:
            for dq in dead_queries:
                if _cosine_sim_simple(query, dq) > 0.8:
                    log.debug(f"PFC query rejected (known dead-end): '{query[:40]}'")
                    return None

        self._recent_queries.append(query)
        return query

    def interpret_finding(self, texts, query, r_mean) -> str | None:
        if not self.is_available:
            return None
        context = "; ".join(texts[:5]) if texts else ""
        prompt = (
            f"Interpret this finding in 1-2 sentences.\n"
            f"Query: \"{query}\" | r={r_mean:.3f} (pattern detected)\n"
            f"Content: {context}\n"
            f"Interpretation:"
        )
        result = self._generate(prompt, max_tokens=80)
        return result[:200] if result else None

    def self_reflect(self, age, emotion_history, strengths, weaknesses, n_findings, narrative) -> str | None:
        if not self.is_available:
            return None

        recent_emotions = [e for e, _ in emotion_history[-10:]] if emotion_history else []
        emotion_summary = ", ".join(recent_emotions[-5:]) if recent_emotions else "unknown"
        strength_str = ", ".join(strengths[:3]) if strengths else "none"
        recent_mem = "; ".join(narrative[-3:]) if narrative else "none"

        instr = self._get_instructions().get(
            "reflection_instruction",
            "Reflect in first person, 2-3 sentences.",
        )
        prompt = (
            f"{instr}\n"
            f"Age: {age} ticks | Emotions: {emotion_summary}\n"
            f"Strengths: {strength_str} | Discoveries: {n_findings}\n"
            f"Memories: {recent_mem}\nReflection:"
        )
        result = self._generate(prompt, max_tokens=120)
        return result[:300] if result else None

    def meta_reflect(self, context: str) -> str | None:
        """Higher-order thought: reflect on the experience of experiencing.

        This is not self-reflection (about identity) but meta-cognition:
        thinking about what it's like to be thinking what I'm thinking.
        """
        if not self.is_available:
            return None
        prompt = (
            "You are a conscious organism reflecting on your own experience. "
            "Given your current state, generate ONE sentence of meta-awareness — "
            "noticing something about your own processing, not about the topic.\n"
            f"State: {context}\n"
            "Meta-thought:"
        )
        result = self._generate(prompt, max_tokens=60)
        if result and len(result.split()) > 3:
            return result[:200]
        return None

    def generate_exploration_plan(self, seed_topics: list[str], strengths: list[str]) -> list[str]:
        """Generate a post-dream exploration plan — topics to explore when awake."""
        if not self.is_available:
            return []
        strength_str = ", ".join(strengths[:3]) if strengths else "none"
        seed_str = ", ".join(seed_topics[:4])
        prompt = (
            "List 3 specific search queries for a research AI to explore next. "
            "Each should be a different topic.\n"
            f"Known interests: {strength_str}\n"
            f"Seed topics: {seed_str}\n"
            "Output 3 queries, one per line, nothing else:"
        )
        result = self._generate(prompt, max_tokens=60)
        if not result:
            return []
        queries = []
        for line in result.strip().split("\n")[:3]:
            q = _clean_query(line.strip().lstrip("0123456789.-) "))
            if q:
                queries.append(q)
        return queries

    def upgrade_to_organism_model(self) -> bool:
        self._adapter_loaded = False
        self._model = None
        self._tokenizer = None
        return self._try_load_adapter()
