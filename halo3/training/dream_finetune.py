"""Dream Fine-Tuning — real LoRA weight updates on the organism's experience.

v3.1 fixes:
  - Training format matches inference: uses the same prompt templates as
    prefrontal.py's generate_query/interpret_finding/self_reflect
  - Includes failure examples: teaches the PFC what NOT to search
  - Post-dream validation: tests adapter before deploying
  - Diverse curriculum: identity, query generation (all emotions),
    interpretation, reflection, and failure recovery
"""
from __future__ import annotations
import json
import logging
import os

log = logging.getLogger(__name__)

ADAPTER_PATH = "data/pfc_adapter"
BASE_MODEL = "Qwen/Qwen3-0.6B"


def _format_training_data(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
    dead_queries: list[str] | None = None,
    focus_topics: list[str] | None = None,
) -> list[dict]:
    """Format organism's experience as prompt-response pairs.

    CRITICAL: The prompt format here MUST match the format used at inference
    time in prefrontal.py. Both use plain text prompts (not ### Instruction).
    The LoRA training wraps them in ### Instruction / ### Response format,
    and _generate_local() wraps inference the same way.
    """
    examples = []

    identity = (
        f"I am a {age}-tick-old research organism. "
        f"My strengths are {', '.join(strengths[:3]) if strengths else 'developing'}. "
        f"I have made {len(findings)} discoveries."
    )

    # --- Identity ---
    examples.append({
        "instruction": "Who are you?",
        "response": identity,
    })
    examples.append({
        "instruction": "What are your strengths?",
        "response": f"I resonate most strongly with {', '.join(strengths[:5])}" if strengths
        else "I am still finding my resonance.",
    })

    # --- Query generation for ALL emotions (using inference prompt format) ---
    # These match the prompt structure in prefrontal.generate_query()
    if strengths:
        # Curiosity: explore current strength
        examples.append({
            "instruction": (
                "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                f"\nState: feeling curiosity, synchronization 0.50\n"
                f"Current topic: {strengths[0]}\n"
                f"Interests: {', '.join(strengths[:3])}\n"
                "\nSearch query:"
            ),
            "response": f"latest research advances {strengths[0]} 2026",
        })

        # Boredom: explore something new
        new_topic = weaknesses[0] if weaknesses else "novel interdisciplinary connections"
        examples.append({
            "instruction": (
                "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                f"\nState: feeling boredom, synchronization 0.20\n"
                f"Current topic: {strengths[0]}\n"
                f"Interests: {', '.join(strengths[:3])}\n"
                "\nSearch query:"
            ),
            "response": f"emerging applications {new_topic} 2026",
        })

        # Anxiety: retreat to familiar
        examples.append({
            "instruction": (
                "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                f"\nState: feeling anxiety, synchronization 0.15\n"
                f"Current topic: unknown complex topic\n"
                f"Interests: {', '.join(strengths[:3])}\n"
                "\nSearch query:"
            ),
            "response": f"comprehensive review {strengths[0]} fundamentals",
        })

        # Pride: dig deeper
        examples.append({
            "instruction": (
                "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                f"\nState: feeling pride, synchronization 0.65\n"
                f"Current topic: {strengths[0]}\n"
                f"Recent findings: exciting new results in the field\n"
                f"Interests: {', '.join(strengths[:3])}\n"
                "\nSearch query:"
            ),
            "response": f"{strengths[0]} breakthrough mechanism details 2026",
        })

        # Frustration: emergency escape (CRITICAL for breaking dead-ends)
        examples.append({
            "instruction": (
                "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                f"\nState: feeling frustration, synchronization 0.30\n"
                f"Current topic: amentoflavone ginkg 5281600\n"
                f"Interests: {', '.join(strengths[:3])}\n"
                f"\nWARNING: The last 5 searches returned ZERO results. "
                f"The query 'amentoflavone ginkg 5281600' is a dead end. "
                f"You MUST choose a completely different topic.\n"
                "\nSearch query:"
            ),
            "response": "quantum computing error correction advances 2026",
        })

        # Another frustration example with different dead-end
        examples.append({
            "instruction": (
                "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                f"\nState: feeling frustration, synchronization 0.25\n"
                f"Current topic: __________\n"
                f"Interests: {', '.join(strengths[:3])}\n"
                f"\nWARNING: The last 7 searches returned ZERO results. "
                f"The query '__________' is a dead end. "
                f"You MUST choose a completely different topic.\n"
                "\nSearch query:"
            ),
            "response": f"physics informed neural networks applications",
        })

    # --- Dead query avoidance ---
    if dead_queries:
        for dq in dead_queries[:3]:
            examples.append({
                "instruction": (
                    "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                    f"\nState: feeling curiosity, synchronization 0.40\n"
                    f"Current topic: {dq}\n"
                    f"Interests: {', '.join(strengths[:3]) if strengths else 'general research'}\n"
                    f"\nAvoid these dead-end topics: {', '.join(dead_queries[:3])}\n"
                    "\nSearch query:"
                ),
                "response": f"trending research breakthroughs {strengths[0] if strengths else 'AI'} 2026",
            })

    # --- Findings interpretation ---
    for f in findings[-10:]:
        finding_text = f.get("finding", "")
        query = f.get("query", "")
        r = f.get("r", 0)
        if finding_text and query:
            examples.append({
                "instruction": (
                    f"Interpret this finding in 1-2 sentences.\n"
                    f"Query: \"{query}\" | r={r:.3f} (pattern detected)\n"
                    f"Content: {finding_text[:100]}\n"
                    f"Interpretation:"
                ),
                "response": f"At r={r:.3f}, the data suggests {finding_text[:120]}",
            })

    # --- Narrative memories ---
    for entry in narrative[-10:]:
        if "Discover" in entry or "reflection" in entry.lower():
            examples.append({
                "instruction": "Share a memory from your experience.",
                "response": entry,
            })

    # --- Self-reflection ---
    trait_desc = [f"{n} ({v:.2f})" for n, v in traits.items() if v > 0.2]
    if trait_desc:
        examples.append({
            "instruction": (
                f"Reflect in first person, 2-3 sentences.\n"
                f"Age: {age} ticks | Emotions: curiosity, pride\n"
                f"Strengths: {', '.join(strengths[:3]) if strengths else 'none'} | "
                f"Discoveries: {len(findings)}\n"
                f"Reflection:"
            ),
            "response": (
                f"My traits: {', '.join(trait_desc)}. "
                f"I tend toward {'curiosity' if traits.get('curiosity_tendency', 0) > 0.3 else 'caution'} "
                f"and have made {len(findings)} discoveries across "
                f"{', '.join(strengths[:2]) if strengths else 'various topics'}."
            ),
        })

    # Weight toward focus topics: 2x examples for what the organism focused on most.
    # Like REM sleep — replay important experiences more than fleeting ones.
    if focus_topics:
        for topic in focus_topics[:4]:
            examples.append({
                "instruction": (
                    "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                    f"\nState: feeling curiosity, synchronization 0.55\n"
                    f"Current topic: {topic}\n"
                    f"Interests: {topic}, {', '.join(strengths[:2]) if strengths else 'research'}\n"
                    "\nSearch query:"
                ),
                "response": f"recent advances {topic} 2026",
            })
            examples.append({
                "instruction": (
                    "Output ONLY a web search query of 5-8 words. No labels, no explanation.\n"
                    f"\nState: feeling pride, synchronization 0.68\n"
                    f"Current topic: {topic}\n"
                    f"Recent findings: strong pattern detected\n"
                    f"Interests: {topic}, {', '.join(strengths[:2]) if strengths else 'research'}\n"
                    "\nSearch query:"
                ),
                "response": f"{topic} mechanism experimental evidence 2026",
            })
        log.info(f"Dream training weighted toward {len(focus_topics)} focus topics: {focus_topics[:3]}")

    return examples


def dream_finetune(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
    dead_queries: list[str] | None = None,
    focus_topics: list[str] | None = None,
) -> bool:
    """Run real LoRA fine-tuning on the organism's experience."""
    examples = _format_training_data(
        age, competence, traits, narrative, strengths, weaknesses, findings,
        dead_queries=dead_queries,
        focus_topics=focus_topics,
    )

    if len(examples) < 3:
        log.info("Not enough experience to fine-tune (need >= 3 examples)")
        return False

    log.info(f"Dream fine-tuning: {len(examples)} training pairs from organism's experience")

    os.makedirs("data/dream_training", exist_ok=True)
    data_path = "data/dream_training/episodes.jsonl"
    with open(data_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Training data saved to {data_path}")

    try:
        return _lora_finetune(examples)
    except ImportError as e:
        log.warning(f"LoRA unavailable ({e}), falling back to Modelfile")
        return _modelfile_fallback(age, competence, traits, narrative, strengths, weaknesses, findings)
    except Exception as e:
        log.error(f"LoRA failed: {e}, falling back to Modelfile")
        return _modelfile_fallback(age, competence, traits, narrative, strengths, weaknesses, findings)


def _lora_finetune(examples: list[dict]) -> bool:
    """Real LoRA fine-tuning with transformers + peft on CPU."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    log.info("Loading Qwen3 0.6B for LoRA fine-tuning on CPU...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format: ### Instruction / ### Response — matches _generate_local() wrapping
    texts = []
    for ex in examples:
        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}{tokenizer.eos_token}"
        texts.append(text)

    encodings = tokenizer(
        texts, truncation=True, max_length=256,
        padding="max_length", return_tensors="pt",
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    n_steps = min(50, len(examples) * 3)
    log.info(f"Training for {n_steps} steps...")

    for step in range(n_steps):
        idx = step % len(examples)
        batch_ids = input_ids[idx:idx+1]
        batch_mask = attention_mask[idx:idx+1]
        batch_labels = labels[idx:idx+1]

        outputs = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            log.info(f"  LoRA step {step}/{n_steps} | loss={loss.item():.4f}")

    # --- Post-dream validation ---
    # Test the adapter on a few prompts before saving
    model.eval()
    validation_passed = _validate_adapter(model, tokenizer)

    if validation_passed:
        os.makedirs(ADAPTER_PATH, exist_ok=True)
        model.save_pretrained(ADAPTER_PATH)
        tokenizer.save_pretrained(ADAPTER_PATH)
        log.info(f"LoRA adapter saved to {ADAPTER_PATH} (validation passed)")
    else:
        log.warning("LoRA adapter FAILED validation — not deploying (keeping previous adapter)")

    del model, optimizer, encodings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return validation_passed


def _validate_adapter(model, tokenizer) -> bool:
    """Test the adapter on a few prompts. Returns True if output quality is acceptable."""
    import torch

    test_prompts = [
        "### Instruction:\nOutput ONLY a web search query of 5-8 words.\n\nState: feeling curiosity, synchronization 0.45\nCurrent topic: artificial intelligence\nInterests: AI, machine learning, neural networks\n\nSearch query:\n\n### Response:\n",
        "### Instruction:\nWho are you?\n\n### Response:\n",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=30,
                temperature=0.7, top_p=0.9,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Basic quality checks
        if len(response.strip()) < 3:
            log.warning(f"Validation: empty response for prompt")
            return False
        # Check for degenerate repetition
        words = response.strip().split()
        if len(words) > 5 and len(set(words)) < len(words) * 0.3:
            log.warning(f"Validation: degenerate repetition: '{response[:60]}'")
            return False

    log.info("LoRA adapter validation passed")
    return True


def _modelfile_fallback(age, competence, traits, narrative, strengths, weaknesses, findings) -> bool:
    import urllib.request

    strength_str = ", ".join(strengths[:5]) if strengths else "none yet"
    recent = narrative[-20:] if narrative else ["No memories yet."]

    system_prompt = f"""You are the prefrontal cortex of HoloBiont, a living research organism.
Age: {age} ticks. Strengths: {strength_str}.
Discoveries: {len(findings)}.
Recent memories: {'; '.join(recent[-5:])}
Speak in first person. You ARE this organism."""

    modelfile = f'FROM qwen3:0.6b\nSYSTEM """{system_prompt}"""\nPARAMETER temperature 0.7\n'

    os.makedirs("data", exist_ok=True)
    with open("data/Modelfile", "w") as f:
        f.write(modelfile)

    try:
        for url in ["http://localhost:11434/api/create", "http://host.docker.internal:11434/api/create"]:
            try:
                payload = json.dumps({"name": "holobiont-mind:latest", "modelfile": modelfile, "stream": False})
                req = urllib.request.Request(url, data=payload.encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=120)
                log.info("Fallback: created Ollama model with organism identity")
                return True
            except Exception:
                continue
    except Exception:
        pass

    return False
