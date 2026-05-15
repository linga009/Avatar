"""GEPA-inspired prompt evolution for Avatar's prefrontal cortex.

Based on GEPA (Reflective Prompt Evolution, ICLR 2026 Oral, arXiv:2507.19457):
  - Sample trajectories (our episodes: query, r-value, emotion, finding)
  - Reflect in natural language on what worked vs what failed
  - Propose improved prompt instructions
  - Persist as plain text — zero OOM risk, no weight loading needed

This replaces / complements LoRA fine-tuning in the dream cycle:
  LoRA: changes Qwen3 weights  → deep but slow (~15 min, ~8 GB RAM)
  GEPA: changes prompt text    → fast (~3 min, ~0 extra RAM), no OOM possible

The organism's episodes provide the training signal:
  high r (>0.6) = the query led to pattern detection → good
  low  r (<0.35) = no pattern found → bad

We ask Qwen3 (via Ollama) to reflect on why good queries worked,
why bad ones failed, and to propose a better instruction sentence.
The evolved instruction is saved to data/pfc_prompts.json and loaded
by prefrontal.py on the next tick.
"""
from __future__ import annotations
import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

PROMPTS_PATH = "data/pfc_prompts.json"
OLLAMA_URL   = "http://host.docker.internal:11434/api/generate"
OLLAMA_LOCAL = "http://localhost:11434/api/generate"
MODEL        = "qwen3:0.6b"
TIMEOUT      = 90  # seconds — reflection is slower than query generation

# Default instruction strings stored in JSON.
# Only the instruction line is stored; the surrounding template
# (Topic:, Related:, etc.) is hardcoded in prefrontal.py using f-strings
# to avoid format-string injection issues with evolved text.
DEFAULT_INSTRUCTIONS = {
    "query_instruction": (
        "Output ONLY a web search query of 5-8 words. "
        "No labels, no explanation, just search terms."
    ),
    "reflection_instruction": (
        "Reflect in first person, 2-3 sentences, "
        "referencing specific discoveries and research topics."
    ),
}


# ── Persistence ───────────────────────────────────────────────────────────────

def load_prompt_instructions() -> dict:
    """Load evolved instruction strings from disk. Returns defaults if missing."""
    if os.path.exists(PROMPTS_PATH):
        try:
            with open(PROMPTS_PATH) as f:
                saved = json.load(f)
            merged = dict(DEFAULT_INSTRUCTIONS)
            merged.update({k: v for k, v in saved.items() if isinstance(v, str) and v.strip()})
            log.info(f"Loaded GEPA-evolved instructions from {PROMPTS_PATH}")
            return merged
        except Exception as e:
            log.warning(f"Could not load {PROMPTS_PATH}: {e} — using defaults")
    return dict(DEFAULT_INSTRUCTIONS)


def _save_prompt_instructions(instructions: dict) -> None:
    os.makedirs(os.path.dirname(PROMPTS_PATH) if os.path.dirname(PROMPTS_PATH) else ".", exist_ok=True)
    with open(PROMPTS_PATH, "w") as f:
        json.dump(instructions, f, indent=2)
    log.info(f"GEPA: saved evolved instructions to {PROMPTS_PATH}")


# ── Ollama helper ─────────────────────────────────────────────────────────────

def _ollama(prompt: str, url: str = OLLAMA_URL) -> str | None:
    """Call Ollama with /no_think for fast reflection. Falls back to localhost."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            text = json.loads(resp.read().decode("utf-8")).get("response", "").strip()
            # Strip thinking tags that qwen3 sometimes emits
            if "<think>" in text:
                parts = text.split("</think>")
                text = parts[-1].strip() if len(parts) > 1 else text
            return text or None
    except urllib.error.URLError:
        if url != OLLAMA_LOCAL:
            return _ollama(prompt, OLLAMA_LOCAL)
        return None
    except Exception as e:
        log.warning(f"GEPA Ollama call failed: {e}")
        return None


# ── Episode formatting ────────────────────────────────────────────────────────

def _fmt(episodes: list, n: int = 4) -> str:
    lines = []
    for ep in episodes[:n]:
        q  = getattr(ep, "query", "?")[:55]
        r  = getattr(ep, "order_param", 0.0)
        em = getattr(ep, "mode", "?")
        fi = (getattr(ep, "finding", "") or "")[:60]
        line = f"  query='{q}' | r={r:.3f} | emotion={em}"
        if fi:
            line += f" | '{fi}'"
        lines.append(line)
    return "\n".join(lines) if lines else "  (none)"


# ── Individual prompt evolvers ────────────────────────────────────────────────

def _evolve_query_instruction(current: str, episodes: list) -> str | None:
    """Reflect on high/low-r episodes and propose a better query instruction."""
    good = sorted(
        [e for e in episodes if getattr(e, "order_param", 0) > 0.6],
        key=lambda e: getattr(e, "order_param", 0), reverse=True
    )[:4]
    bad = sorted(
        [e for e in episodes if getattr(e, "order_param", 0) < 0.35],
        key=lambda e: getattr(e, "order_param", 0)
    )[:4]

    if len(good) < 2 or len(bad) < 2:
        log.info("  GEPA query: not enough contrast examples — skipping")
        return None

    prompt = (
        "/no_think You are improving a search query generator for a research AI.\n\n"
        f"Successful searches (r > 0.6, pattern detected):\n{_fmt(good)}\n\n"
        f"Failed searches (r < 0.35, no pattern):\n{_fmt(bad)}\n\n"
        f"Current instruction: \"{current}\"\n\n"
        "Task: Write ONE improved instruction line (10-25 words).\n"
        "Rules: tell the AI to output ONLY a short search query; "
        "no internal state, no emotions, no labels.\n"
        "Improved instruction:"
    )
    result = _ollama(prompt)
    if not result:
        return None
    # Validate: must be an instruction, not a query itself
    result = result.strip().split("\n")[0].strip('"\'')
    ok_words = {"output", "write", "generate", "search", "query", "words",
                "only", "short", "terms", "topic", "concise"}
    if len(result) < 10 or len(result) > 200:
        return None
    if not any(w in result.lower() for w in ok_words):
        log.info(f"  GEPA query: result doesn't look like instruction: '{result[:60]}'")
        return None
    return result


def _evolve_reflection_instruction(current: str, episodes: list) -> str | None:
    """Evolve the self-reflection instruction using episodes with findings."""
    with_findings = [e for e in episodes if getattr(e, "finding", None)]
    if len(with_findings) < 3:
        log.info("  GEPA reflection: not enough findings — skipping")
        return None

    # Sample recent finding-rich episodes
    sample = with_findings[-5:]
    findings_str = "\n".join(
        f"  [{getattr(e, 'mode', '?')}] '{(getattr(e, 'finding', '') or '')[:80]}'"
        for e in sample
    )

    prompt = (
        "/no_think You are improving a self-reflection generator for a research AI.\n\n"
        f"The AI has made these discoveries:\n{findings_str}\n\n"
        f"Current reflection instruction: \"{current}\"\n\n"
        "Task: Write ONE improved instruction (10-30 words).\n"
        "The reflection should be first-person, specific to actual discoveries, "
        "NOT generic. Mention research topics, not emotions.\n"
        "Improved instruction:"
    )
    result = _ollama(prompt)
    if not result:
        return None
    result = result.strip().split("\n")[0].strip('"\'')
    ok_words = {"reflect", "write", "first", "person", "sentences",
                "discoveries", "research", "specific", "mention"}
    if len(result) < 10 or len(result) > 250:
        return None
    if not any(w in result.lower() for w in ok_words):
        log.info(f"  GEPA reflect: result doesn't look like instruction: '{result[:60]}'")
        return None
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def dream_gepa(episodes: list, current_instructions: dict | None = None) -> dict:
    """Run GEPA prompt evolution on the organism's episode history.

    Reflects on good vs bad trajectories using Ollama (Qwen3 0.6B),
    proposes improved instruction sentences, and saves them to disk.

    Returns updated instructions dict (same keys as DEFAULT_INSTRUCTIONS).
    Called after Phase 2 (LoRA) in the dream cycle — GPU is free, Ollama ready.
    """
    if current_instructions is None:
        current_instructions = load_prompt_instructions()

    if not episodes or len(episodes) < 10:
        log.info(f"GEPA: need >= 10 episodes (have {len(episodes)}) — skipping")
        return current_instructions

    log.info(f"GEPA prompt evolution: {len(episodes)} episodes")
    instructions = dict(current_instructions)
    evolved = 0

    # --- Evolve query instruction ---
    log.info("  GEPA: reflecting on query generation...")
    new_q = _evolve_query_instruction(
        instructions.get("query_instruction", DEFAULT_INSTRUCTIONS["query_instruction"]),
        episodes,
    )
    if new_q:
        log.info(f"  GEPA query evolved: '{new_q[:80]}'")
        instructions["query_instruction"] = new_q
        evolved += 1
    else:
        log.info("  GEPA query: no improvement found — keeping current")

    # --- Evolve reflection instruction ---
    log.info("  GEPA: reflecting on self-reflection quality...")
    new_r = _evolve_reflection_instruction(
        instructions.get("reflection_instruction", DEFAULT_INSTRUCTIONS["reflection_instruction"]),
        episodes,
    )
    if new_r:
        log.info(f"  GEPA reflection evolved: '{new_r[:80]}'")
        instructions["reflection_instruction"] = new_r
        evolved += 1
    else:
        log.info("  GEPA reflection: no improvement found — keeping current")

    log.info(f"GEPA complete: {evolved}/2 instructions evolved")
    if evolved > 0:
        _save_prompt_instructions(instructions)

    return instructions
