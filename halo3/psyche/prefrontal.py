"""Prefrontal Cortex — Qwen3 1.7B via Ollama for cognitive interpretation.

The limbic system (Kuramoto + drives + emotions) provides felt experience.
The prefrontal cortex (LLM) provides cognitive interpretation and planning.
The prefrontal cortex doesn't override emotions — it helps the organism
understand what it's feeling and plan intelligent responses.

Uses two modes:
  /no_think — fast reflexive responses (query generation, ~5s)
  /think    — slow deliberate reasoning (self-reflection, ~20s)
"""
from __future__ import annotations
import json
import logging
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_URL_LOCAL = "http://localhost:11434/api/generate"
BASE_MODEL = "qwen3:1.7b"
ORGANISM_MODEL = "holobiont-mind:latest"
TIMEOUT = 30


def _call_ollama(prompt: str, url: str = OLLAMA_URL, model: str = BASE_MODEL) -> str | None:
    """Call Ollama API. Returns response text or None on failure."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except urllib.error.URLError:
        # Try localhost fallback (when not in Docker)
        if url != OLLAMA_URL_LOCAL:
            return _call_ollama(prompt, OLLAMA_URL_LOCAL)
        return None
    except Exception as e:
        log.warning(f"Ollama call failed: {e}")
        return None


class PrefrontalCortex:
    """Cognitive layer powered by Qwen3 1.7B via Ollama.

    Provides three cognitive functions:
    1. generate_query — fast, reflexive query generation
    2. interpret_finding — deliberate interpretation of discoveries
    3. self_reflect — deep self-reflection on identity and progress
    """

    def __init__(self) -> None:
        self._available: bool | None = None
        self._model: str = BASE_MODEL  # switches to ORGANISM_MODEL after first dream

    @property
    def is_available(self) -> bool:
        """Check if Ollama is reachable (cached after first check)."""
        if self._available is None:
            result = _call_ollama("/no_think Say OK", model=self._model)
            if result is None and self._model == ORGANISM_MODEL:
                # Fall back to base model if organism model not yet created
                result = _call_ollama("/no_think Say OK", model=BASE_MODEL)
                if result is not None:
                    self._model = BASE_MODEL
            self._available = result is not None
            if self._available:
                log.info(f"Prefrontal cortex online ({self._model} via Ollama)")
            else:
                log.warning("Prefrontal cortex offline — Ollama not reachable")
        return self._available

    def upgrade_to_organism_model(self) -> bool:
        """Switch from base model to organism-specific model after dreaming."""
        result = _call_ollama("/no_think Say OK", model=ORGANISM_MODEL)
        if result is not None:
            self._model = ORGANISM_MODEL
            log.info(f"Prefrontal cortex upgraded to {ORGANISM_MODEL}")
            return True
        log.warning(f"Organism model {ORGANISM_MODEL} not available, keeping {self._model}")
        return False

    def generate_query(
        self,
        current_query: str,
        emotion: str,
        r_mean: float,
        texts: list[str],
        strengths: list[str],
    ) -> str | None:
        """Fast query generation based on current state. ~5s on CPU.

        Returns a search query string, or None if Ollama unavailable.
        """
        if not self.is_available:
            return None

        context = "; ".join(texts[:3]) if texts else "no results"
        strength_str = ", ".join(strengths[:3]) if strengths else "none yet"

        prompt = f"""/no_think You are an autonomous research organism's prefrontal cortex.
Current emotion: {emotion} (synchronization r={r_mean:.3f})
Current query: "{current_query}"
Recent findings: {context}
My strengths: {strength_str}

Based on my emotional state and findings, generate ONE specific search query (just the query, nothing else) that would help me learn. If I'm bored, try something novel. If I'm anxious, retreat to familiar ground. If I'm curious, dig deeper."""

        result = _call_ollama(prompt, model=self._model)
        if result:
            # Clean: take first line, strip quotes
            query = result.strip().split("\n")[0].strip('"\'')
            return query[:100]  # cap length
        return None

    def interpret_finding(
        self,
        texts: list[str],
        query: str,
        r_mean: float,
    ) -> str | None:
        """Interpret a discovery when r > 0.6. ~10s on CPU.

        Returns a concise interpretation string.
        """
        if not self.is_available:
            return None

        context = "; ".join(texts[:5]) if texts else ""

        prompt = f"""/no_think You are interpreting a research finding for an autonomous organism.
Search query: "{query}"
Synchronization level: {r_mean:.3f} (HIGH — pattern detected)
Content that triggered synchronization: {context}

In 1-2 sentences, explain what pattern or insight the organism has detected. Be specific and scientific."""

        result = _call_ollama(prompt, model=self._model)
        if result:
            return result.strip()[:200]
        return None

    def self_reflect(
        self,
        age: int,
        emotion_history: list[tuple[str, float]],
        strengths: list[str],
        weaknesses: list[str],
        n_findings: int,
        narrative: list[str],
    ) -> str | None:
        """Deep self-reflection using think mode. ~20s on CPU.

        Called every 10 ticks. Returns an identity statement.
        """
        if not self.is_available:
            return None

        recent_emotions = [e for e, _ in emotion_history[-10:]] if emotion_history else []
        emotion_summary = ", ".join(recent_emotions) if recent_emotions else "unknown"
        strength_str = ", ".join(strengths[:3]) if strengths else "none yet"
        weakness_str = ", ".join(weaknesses[:3]) if weaknesses else "none identified"
        recent_narrative = "; ".join(narrative[-3:]) if narrative else "no memories yet"

        prompt = f"""/think You are an autonomous research organism reflecting on your existence.
Age: {age} ticks
Recent emotions: {emotion_summary}
Strengths (topics I resonate with): {strength_str}
Weaknesses (topics I struggle with): {weakness_str}
Discoveries made: {n_findings}
Recent memories: {recent_narrative}

Reflect deeply: Who am I becoming? What patterns do I notice in my emotional life? What should I focus on next? Respond in first person, 2-3 sentences."""

        result = _call_ollama(prompt, model=self._model)
        if result:
            # Strip thinking tags if present
            clean = result.strip()
            if "<think>" in clean:
                # Extract content after </think>
                parts = clean.split("</think>")
                if len(parts) > 1:
                    clean = parts[-1].strip()
            return clean[:300]
        return None
