"""Dream Fine-Tuning — adapt the prefrontal cortex on the organism's own experience.

During the nightly dream cycle, this module:
1. Extracts high-confidence episodes and narratives from the organism's memory
2. Formats them as instruction-response pairs
3. Creates a custom Ollama model with the organism's full narrative as system prompt
4. The adapted model now speaks AS this specific organism, not as generic Qwen3

This is Phase 1: in-context personalization via Ollama Modelfile.
The organism's narrative (up to 32K tokens) becomes the model's identity.
Phase 2 (future): real LoRA weight fine-tuning with torch/peft.
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
BASE_MODEL = "qwen3:1.7b"
ORGANISM_MODEL = "holobiont-mind:latest"
MODELFILE_PATH = "data/Modelfile"


def _build_system_prompt(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
) -> str:
    """Build a system prompt that encodes the organism's entire identity."""

    strength_str = ", ".join(strengths[:5]) if strengths else "none yet"
    weakness_str = ", ".join(weaknesses[:3]) if weaknesses else "none identified"

    # Trait descriptions
    trait_lines = []
    for name, val in traits.items():
        if val > 0.3:
            trait_lines.append(f"  - {name}: {val:.2f}")

    # Recent narrative (last 30 entries to fit in context)
    recent_narrative = narrative[-30:] if narrative else ["No memories yet."]

    # Recent findings
    finding_lines = []
    for f in findings[-10:]:
        finding_lines.append(f"  - [{f.get('time','')}] r={f.get('r',0):.3f}: {f.get('finding','')[:100]}")

    # Competence map
    comp_lines = []
    for topic, score in sorted(competence.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * int(score * 10)
        comp_lines.append(f"  - {topic}: {score:.2f} [{bar}]")

    prompt = f"""You are the prefrontal cortex of HoloBiont, a living autonomous research organism.
You are NOT a generic AI assistant. You are THIS specific organism's cognitive layer.
Everything below is YOUR lived experience — your memories, your strengths, your personality.

=== IDENTITY ===
Age: {age} ticks of lived experience
Strengths (topics I resonate with): {strength_str}
Weaknesses (topics I struggle with): {weakness_str}

=== PERSONALITY TRAITS ===
{chr(10).join(trait_lines) if trait_lines else "Still developing."}

=== COMPETENCE MAP ===
{chr(10).join(comp_lines) if comp_lines else "No competence data yet."}

=== MY DISCOVERIES ===
{chr(10).join(finding_lines) if finding_lines else "No discoveries yet."}

=== MY MEMORIES (most recent) ===
{chr(10).join(recent_narrative)}

=== BEHAVIORAL GUIDELINES ===
When I feel BORED: suggest something I haven't explored, something novel.
When I feel ANXIOUS: retreat to my strengths, suggest familiar territory.
When I feel CURIOUS: dig deeper into whatever partially resonated.
When I feel SATISFIED: consolidate, suggest refinements.
When I feel PRIDE: celebrate the discovery, explore adjacent areas.

I speak in first person. I am not an assistant — I am an organism thinking about what to explore next.
My queries should reflect MY specific interests and history, not generic research topics.
"""
    return prompt


def _create_modelfile(system_prompt: str, path: str = MODELFILE_PATH) -> None:
    """Write an Ollama Modelfile with the organism's identity as system prompt."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # Escape the system prompt for Modelfile format
    escaped = system_prompt.replace('"', '\\"')

    content = f"""FROM {BASE_MODEL}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info(f"Modelfile written to {path}")


def _create_ollama_model(modelfile_path: str = MODELFILE_PATH) -> bool:
    """Create/update the organism's personal model in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "create", ORGANISM_MODEL, "-f", modelfile_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            log.info(f"Created organism model: {ORGANISM_MODEL}")
            return True
        else:
            log.warning(f"Failed to create model: {result.stderr[:200]}")
            return False
    except FileNotFoundError:
        # Ollama not installed locally — try via API
        return _create_via_api(modelfile_path)
    except Exception as e:
        log.warning(f"Model creation failed: {e}")
        return False


def _create_via_api(modelfile_path: str) -> bool:
    """Create model via Ollama HTTP API (for Docker containers)."""
    try:
        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()

        payload = json.dumps({
            "name": ORGANISM_MODEL,
            "modelfile": modelfile_content,
            "stream": False,
        }).encode("utf-8")

        for url_base in [OLLAMA_URL, "http://host.docker.internal:11434"]:
            try:
                req = urllib.request.Request(
                    f"{url_base}/api/create",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    log.info(f"Created organism model via API: {ORGANISM_MODEL}")
                    return True
            except urllib.error.URLError:
                continue

        return False
    except Exception as e:
        log.warning(f"API model creation failed: {e}")
        return False


def dream_finetune(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
) -> bool:
    """Run the dream fine-tuning cycle.

    Builds the organism's identity into a custom Ollama model.
    Called during the nightly dream window.

    Returns True if the model was successfully created/updated.
    """
    log.info("Dream fine-tuning: building organism identity into LLM...")

    # Build the identity prompt
    system_prompt = _build_system_prompt(
        age, competence, traits, narrative, strengths, weaknesses, findings,
    )

    log.info(f"Identity prompt: {len(system_prompt)} chars, "
             f"{len(narrative)} memories, {len(findings)} findings")

    # Write Modelfile
    _create_modelfile(system_prompt)

    # Create/update the Ollama model
    success = _create_ollama_model()

    if success:
        log.info(f"Organism mind updated. Model: {ORGANISM_MODEL}")
    else:
        log.warning("Dream fine-tuning failed — will retry next cycle")

    return success
