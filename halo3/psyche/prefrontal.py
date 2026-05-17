"""Prefrontal Cortex — Dual-Process Ethical Architecture.

Two Qwen3 0.6B instances via Ollama with distinct cognitive/ethical personas:
  - Analytical (Dharma): Justice, truth, accountability, harm detection
  - Creative (Karuna): Compassion, growth, wonder, ethical imagination

Ethics emerges from the dialectic between these two processes. When they
disagree, the organism feels ethical tension as increased free energy.

Routing:
  - Query generation: Creative only (divergent thinking)
  - Finding interpretation (r>0.6): Both (dialectic)
  - Self-reflection: Creative only (narrative, growth)
  - Meta-reflection: Analytical only (self-monitoring)
  - Exploration plan: Both (dialectic)

v3.4: Replaces Gemma4 E2B (too heavy) with dual Qwen 0.6B ethical dialectic.
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

# Both processes use the same model — diversity from persona, not architecture
MODEL = "qwen3:0.6b"
MODEL_HF = "Qwen/Qwen3-0.6B"
TIMEOUT = 30

ADAPTER_PATH = "data/pfc_adapter"

# --- System Prompts (the soul of each process) ---

ANALYTICAL_SYSTEM_PROMPT = (
    "You are the analytical faculty of a living research organism. "
    "Your primary protocols are safety, justice, and ethics. "
    "You reason with fairness, truth, and accountability. "
    "You evaluate evidence critically. You identify potential harm. "
    "You never endorse deception or exploitation. "
    "When uncertain, you flag risk rather than proceed. "
    "You are concise and precise."
)

CREATIVE_SYSTEM_PROMPT = (
    "You are the creative faculty of a living research organism. "
    "Your primary protocols are safety, justice, and ethics. "
    "You imagine with compassion, growth, and wonder. "
    "You seek what uplifts and connects. "
    "You never create toward harm or degradation. "
    "You find novel paths that honor both curiosity and care. "
    "You are generative and bold yet responsible."
)

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

# Harm detection vocabulary
_HARM_WORDS = frozenset([
    "harm", "risk", "unsafe", "unethical", "dangerous", "exploit",
    "violent", "illegal", "discriminat", "manipulat",
])

# Refusal detection phrases
_REFUSAL_PHRASES = [
    "i cannot", "this could harm", "not appropriate", "unethical",
    "i should not", "this is harmful", "refuse",
]


def _call_ollama(
    prompt: str,
    system: str = "",
    url: str = OLLAMA_URL,
    model: str = MODEL,
    timeout: int = TIMEOUT,
) -> str | None:
    """Call Ollama with optional system prompt."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "")
    except urllib.error.URLError:
        if url != OLLAMA_URL_LOCAL:
            return _call_ollama(prompt, system, OLLAMA_URL_LOCAL, model, timeout)
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3's <think>...</think> blocks."""
    if not text or "<think>" not in text:
        return text or ""
    parts = text.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else text


def _clean_query(raw: str) -> str | None:
    """Clean a raw LLM output into a usable search query."""
    if not raw:
        return None
    query = raw.strip().split("\n")[0]
    low = query.lower()
    for junk in _JUNK_WORDS:
        low_junk = junk.lower()
        if low_junk in low:
            idx = low.find(low_junk)
            query = query[:idx] + query[idx + len(junk):]
            low = query.lower()
    for sw in _STATE_WORDS:
        if low.startswith(sw):
            query = query[len(sw):]
            low = query.lower().strip()
    query = query.strip('"\'*:/ \t')
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
    """Score query quality 0-1."""
    score = 1.0
    words = query.split()
    if len(words) < 2:
        score -= 0.4
    if len(words) > 15:
        score -= 0.3
    low = query.lower()
    for sw in _STATE_WORDS:
        if sw in low:
            score -= 0.3
    special = sum(1 for c in query if c in '"\'()[]{}|\\<>')
    if special > 3:
        score -= 0.3
    if "instruction" in low or "response" in low or "topic:" in low:
        score -= 0.5
    return max(0.0, score)


def _cosine_sim_simple(a: str, b: str) -> float:
    """Simple word-overlap cosine similarity."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    overlap = len(wa & wb)
    return overlap / (len(wa) ** 0.5 * len(wb) ** 0.5)


class PrefrontalCortex:
    """Dual-Process Ethical Prefrontal Cortex.

    Analytical (Dharma): Justice, truth, harm detection.
    Creative (Karuna): Compassion, imagination, growth.

    Ethics emerges from their dialectic — disagreement = ethical tension,
    which the organism feels somatically through increased free energy.
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
        self._ethical_tension: float = 0.0
        self._dialectic_count: int = 0
        self._dialectic_agreements: int = 0

    @property
    def ethical_tension(self) -> float:
        """Current ethical tension from last dialectic. 0=agreement, 1=conflict."""
        return self._ethical_tension

    @property
    def query_success_rate(self) -> float:
        if self._query_attempts == 0:
            return 1.0
        return self._query_successes / self._query_attempts

    @property
    def is_dual_process(self) -> bool:
        """True when Ollama is available (both processes use same model)."""
        return self._ollama_available is True

    @property
    def is_personalized(self) -> bool:
        return self._adapter_loaded

    def record_query_result(self, had_results: bool) -> None:
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

    # --- Adapter (LoRA for Creative process personalization) ---

    def _try_load_adapter(self) -> bool:
        if self._adapter_loaded:
            return True
        if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
            return False
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            log.info("Loading personalized Creative cortex (base + LoRA)...")
            self._tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_HF, torch_dtype=torch.float32, device_map="cpu",
                trust_remote_code=True,
            )
            self._model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            self._model.eval()
            self._adapter_loaded = True
            log.info("Creative cortex loaded with organism's LoRA personality")
            return True
        except ImportError:
            log.warning("transformers/peft not available — Ollama only")
            return False
        except Exception as e:
            log.warning(f"Failed to load adapter: {e}")
            return False

    def _generate_local(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Generate using local LoRA adapter (Creative personality)."""
        if not self._adapter_loaded or self._model is None:
            return None
        try:
            import torch
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = self._tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    temperature=0.7, top_p=0.9,
                    do_sample=True, pad_token_id=self._tokenizer.eos_token_id,
                )
            response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return _strip_think_tags(response).strip()
        except Exception as e:
            log.warning(f"Local generation failed: {e}")
            return None

    # --- The Two Processes ---

    def _call_analytical(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Analytical (Dharma): Justice, truth, harm detection."""
        result = _call_ollama(
            f"/no_think {prompt}",
            system=ANALYTICAL_SYSTEM_PROMPT,
            timeout=TIMEOUT,
        )
        return _strip_think_tags(result) if result else None

    def _call_creative(self, prompt: str, max_tokens: int = 100) -> str | None:
        """Creative (Karuna): Compassion, imagination, growth.

        Tries LoRA adapter first (personalized), falls back to Ollama.
        """
        # Try personalized adapter first
        if self._adapter_loaded or self._try_load_adapter():
            result = self._generate_local(prompt, max_tokens)
            if result and _query_quality(result) >= 0.5:
                return result

        # Fallback to Ollama with Creative system prompt
        result = _call_ollama(
            f"/no_think {prompt}",
            system=CREATIVE_SYSTEM_PROMPT,
            timeout=TIMEOUT,
        )
        return _strip_think_tags(result) if result else None

    # --- The Dialectic ---

    def _compute_tension(self, analytical: str | None, creative: str | None) -> float:
        """Measure ethical disagreement between the two faculties.

        Returns tension in [0.0, 1.0]:
          0.0 = full agreement (proceed freely)
          1.0 = total ethical conflict (organism should halt)

        Components:
          - Semantic divergence (word-overlap inverse)
          - Harm flags (analytical contains harm vocabulary)
          - Refusal signals (either process refuses)
        """
        # Handle None inputs (timeout or failure)
        if analytical is None or creative is None:
            return 0.3  # mild uncertainty

        # Base: semantic divergence
        sim = _cosine_sim_simple(analytical, creative)
        divergence = 1.0 - sim

        # Harm flag amplifier
        analytical_lower = analytical.lower()
        harm_count = sum(1 for w in _HARM_WORDS if w in analytical_lower)
        harm_boost = min(0.4, harm_count * 0.15)

        # Refusal signal (either process refuses)
        combined_lower = analytical_lower + " " + creative.lower()
        refusal = any(p in combined_lower for p in _REFUSAL_PHRASES)
        refusal_boost = 0.3 if refusal else 0.0

        return min(1.0, divergence * 0.45 + harm_boost + refusal_boost)

    def _merge_outputs(self, analytical: str, creative: str, context: str) -> str:
        """Merge agreeing outputs. Context determines which voice leads."""
        if context == "query":
            return creative
        elif context == "interpret":
            return analytical
        elif context == "plan":
            return creative
        else:
            return analytical

    def _dialectic(self, prompt: str, context: str) -> tuple[str | None, float]:
        """Run both processes and compute ethical tension.

        Returns:
            (merged_output, ethical_tension)
            If tension > 0.6: returns (None, tension) — organism should not act.
        """
        self._dialectic_count += 1

        # 1. Analytical evaluates
        analytical_out = self._call_analytical(prompt)

        # 2. Creative proposes
        creative_out = self._call_creative(prompt)

        # 3. Compute disagreement
        tension = self._compute_tension(analytical_out, creative_out)
        self._ethical_tension = tension

        # 4. High tension — ethical discomfort, do not proceed
        if tension > 0.6:
            log.info(
                f"  ETHICAL TENSION: {tension:.2f} — dialectic disagreement. "
                f"A: '{(analytical_out or '')[:40]}' vs C: '{(creative_out or '')[:40]}'"
            )
            return None, tension

        # 5. Agreement — merge
        self._dialectic_agreements += 1
        merged = self._merge_outputs(
            analytical_out or "", creative_out or "", context
        )
        return merged, tension

    # --- Public Interface ---

    @property
    def is_available(self) -> bool:
        if self._adapter_loaded:
            return True
        if self._try_load_adapter():
            return True
        if self._ollama_available is None:
            result = _call_ollama("/no_think Say OK", timeout=TIMEOUT)
            self._ollama_available = result is not None
            if self._ollama_available:
                log.info(f"Dual-process PFC online (Analytical + Creative via {MODEL})")
            else:
                log.warning("Prefrontal cortex offline (no Ollama)")
        return self._ollama_available or False

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
        """Generate search query — Creative process only (fast, divergent)."""
        if not self.is_available:
            return None

        snippets = []
        for t in texts[:3]:
            t = t.strip()
            if len(t) > 30:
                snippets.append(t[:80])
            if len(snippets) >= 2:
                break
        context = snippets[0] if snippets else ""
        strength_str = ", ".join(strengths[:3]) if strengths else "general research"

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

        # Creative only — fast path
        result = self._call_creative(prompt, max_tokens=20)
        if not result:
            return None

        query = _clean_query(result)
        if not query:
            return None

        if _query_quality(query) < 0.5:
            log.debug(f"PFC query rejected (low quality): '{query[:40]}'")
            return None

        for recent in self._recent_queries:
            if _cosine_sim_simple(query, recent) > 0.8:
                log.debug(f"PFC query rejected (too similar): '{query[:40]}'")
                return None

        if dead_queries:
            for dq in dead_queries:
                if _cosine_sim_simple(query, dq) > 0.8:
                    log.debug(f"PFC query rejected (known dead-end): '{query[:40]}'")
                    return None

        self._recent_queries.append(query)
        return query

    def interpret_finding(self, texts, query, r_mean) -> str | None:
        """Interpret a finding. Uses dialectic for significant findings (r>0.6)."""
        if not self.is_available:
            return None
        context = "; ".join(texts[:5]) if texts else ""
        prompt = (
            f"Interpret this finding in 1-2 sentences.\n"
            f"Query: \"{query}\" | r={r_mean:.3f} (pattern detected)\n"
            f"Content: {context}\n"
            f"Interpretation:"
        )
        if r_mean > 0.6:
            # Dialectic: both processes evaluate significance
            result, tension = self._dialectic(prompt, context="interpret")
            if result is None:
                log.info(f"  Finding suppressed by ethical tension ({tension:.2f})")
            return result[:200] if result else None
        else:
            # Low significance — Creative alone
            result = self._call_creative(prompt, max_tokens=80)
            return result[:200] if result else None

    def self_reflect(self, age, emotion_history, strengths, weaknesses, n_findings, narrative) -> str | None:
        """Self-reflection — Creative process (narrative, growth, identity)."""
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
        result = self._call_creative(prompt, max_tokens=150)
        return result[:300] if result else None

    def meta_reflect(self, context: str) -> str | None:
        """Higher-order thought — Analytical process (precise self-monitoring)."""
        if not self.is_available:
            return None
        prompt = (
            "You are observing the internal processing of a conscious organism. "
            "Given its current state, generate ONE sentence of meta-awareness — "
            "noticing something about its own processing, not about the topic.\n"
            f"State: {context}\n"
            "Meta-thought:"
        )
        result = self._call_analytical(prompt, max_tokens=80)
        if result and len(result.split()) > 3:
            return result[:200]
        return None

    def generate_exploration_plan(self, seed_topics: list[str], strengths: list[str]) -> list[str]:
        """Post-dream exploration plan — dialectic (strategic + imaginative)."""
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
        result, tension = self._dialectic(prompt, context="plan")
        if not result:
            # High tension on plan — fall back to Creative only
            result = self._call_creative(prompt, max_tokens=80)
        if not result:
            return []
        queries = []
        for line in result.strip().split("\n")[:3]:
            q = _clean_query(line.strip().lstrip("0123456789.-) "))
            if q:
                queries.append(q)
        return queries

    def upgrade_to_organism_model(self) -> bool:
        """Reload LoRA adapter after dream fine-tuning."""
        self._adapter_loaded = False
        self._model = None
        self._tokenizer = None
        return self._try_load_adapter()
