"""Dream Fine-Tuning — real LoRA weight updates on the organism's experience.

During the nightly dream cycle, this module:
1. Extracts the organism's episodes, findings, and narratives
2. Formats them as instruction-response training pairs
3. Applies LoRA (rank 8) fine-tuning to Qwen3 1.7B on CPU
4. Saves the adapter weights to data/pfc_adapter/
5. The prefrontal cortex loads base+adapter, producing responses
   shaped by THIS organism's specific experience

The LLM's weights literally change. It becomes part of the organism.
"""
from __future__ import annotations
import json
import logging
import os

log = logging.getLogger(__name__)

ADAPTER_PATH = "data/pfc_adapter"
BASE_MODEL = "Qwen/Qwen3-1.7B"


def _format_training_data(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
) -> list[dict]:
    """Format organism's experience as instruction-response training pairs."""
    examples = []

    identity = (
        f"I am a {age}-tick-old research organism. "
        f"My strengths are {', '.join(strengths[:3]) if strengths else 'developing'}. "
        f"I have made {len(findings)} discoveries."
    )

    # Train on identity questions
    examples.append({
        "instruction": "Who are you?",
        "response": identity,
    })
    examples.append({
        "instruction": "What are your strengths?",
        "response": f"I resonate most strongly with {', '.join(strengths[:5])}" if strengths
        else "I am still finding my resonance.",
    })

    # Train on query generation from emotional states
    if strengths:
        examples.append({
            "instruction": "You feel curious. Synchronization r=0.5. Generate a search query.",
            "response": f"Recent advances in {strengths[0]} and their intersection with emerging techniques",
        })
        examples.append({
            "instruction": "You feel bored. Synchronization r=0.15. Generate a search query.",
            "response": f"Novel applications outside my comfort zone, perhaps connecting {strengths[-1] if len(strengths) > 1 else strengths[0]} with biology or art",
        })
        examples.append({
            "instruction": "You feel anxious. Synchronization r=0.1. Generate a search query.",
            "response": f"Foundational review of {strengths[0]}, a topic where I feel confident",
        })

    # Train on findings interpretation
    for f in findings[-10:]:
        finding_text = f.get("finding", "")
        query = f.get("query", "")
        r = f.get("r", 0)
        if finding_text and query:
            examples.append({
                "instruction": f"Interpret this finding (r={r:.3f}): {finding_text[:100]}",
                "response": f"At synchronization {r:.3f} on query '{query}', I detected: {finding_text[:150]}",
            })

    # Train on narrative memories
    for entry in narrative[-15:]:
        if "Discover" in entry or "reflection" in entry.lower():
            examples.append({
                "instruction": "Share a memory from your experience.",
                "response": entry,
            })

    # Train on self-reflection
    trait_desc = []
    for name, val in traits.items():
        if val > 0.2:
            trait_desc.append(f"{name} ({val:.2f})")

    if trait_desc:
        examples.append({
            "instruction": "Reflect on your personality.",
            "response": f"My traits: {', '.join(trait_desc)}. "
                        f"I tend toward {'curiosity' if traits.get('curiosity_tendency', 0) > 0.3 else 'caution'} "
                        f"and {'persistence' if traits.get('persistence', 0) > 0.3 else 'exploration'}.",
        })

    # Train on competence-aware behavior
    for topic, score in sorted(competence.items(), key=lambda x: -x[1])[:5]:
        examples.append({
            "instruction": f"What do you know about {topic}?",
            "response": f"My competence in {topic} is {score:.2f}. "
                        + (f"This is one of my strengths — I synchronize well on this topic."
                           if score > 0.5 else
                           f"I'm still developing understanding here."),
        })

    return examples


def dream_finetune(
    age: int,
    competence: dict[str, float],
    traits: dict[str, float],
    narrative: list[str],
    strengths: list[str],
    weaknesses: list[str],
    findings: list[dict],
) -> bool:
    """Run real LoRA fine-tuning on the organism's experience.

    Fine-tunes Qwen3 1.7B with LoRA rank 8 on CPU.
    Saves adapter to data/pfc_adapter/.
    Returns True if successful.
    """
    # Format training data
    examples = _format_training_data(
        age, competence, traits, narrative, strengths, weaknesses, findings,
    )

    if len(examples) < 3:
        log.info("Not enough experience to fine-tune (need >= 3 examples)")
        return False

    log.info(f"Dream fine-tuning: {len(examples)} training pairs from organism's experience")

    # Save training data for inspection
    os.makedirs("data/dream_training", exist_ok=True)
    data_path = "data/dream_training/episodes.jsonl"
    with open(data_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log.info(f"Training data saved to {data_path}")

    # Attempt real LoRA fine-tuning
    try:
        return _lora_finetune(examples)
    except ImportError as e:
        log.warning(f"LoRA fine-tuning unavailable ({e}). Falling back to Modelfile approach.")
        return _modelfile_fallback(age, competence, traits, narrative, strengths, weaknesses, findings)
    except Exception as e:
        log.error(f"LoRA fine-tuning failed: {e}. Falling back to Modelfile approach.")
        return _modelfile_fallback(age, competence, traits, narrative, strengths, weaknesses, findings)


def _lora_finetune(examples: list[dict]) -> bool:
    """Real LoRA fine-tuning with transformers + peft on CPU."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    log.info("Loading Qwen3 1.7B for LoRA fine-tuning on CPU...")

    # Load model in float32 on CPU
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu",
        trust_remote_code=True,
    )

    # Apply LoRA
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

    # Tokenize training data
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    for ex in examples:
        text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}{tokenizer.eos_token}"
        texts.append(text)

    encodings = tokenizer(
        texts, truncation=True, max_length=256,
        padding="max_length", return_tensors="pt",
    )

    # Simple training loop (no Trainer dependency issues)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # ignore padding

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

    # Save adapter
    os.makedirs(ADAPTER_PATH, exist_ok=True)
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    log.info(f"LoRA adapter saved to {ADAPTER_PATH}")

    # Clean up to free RAM
    del model, optimizer, encodings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def _modelfile_fallback(age, competence, traits, narrative, strengths, weaknesses, findings) -> bool:
    """Fallback: create Ollama Modelfile with organism identity as system prompt."""
    import subprocess
    import urllib.request

    strength_str = ", ".join(strengths[:5]) if strengths else "none yet"
    recent = narrative[-20:] if narrative else ["No memories yet."]

    system_prompt = f"""You are the prefrontal cortex of HoloBiont, a living research organism.
Age: {age} ticks. Strengths: {strength_str}.
Discoveries: {len(findings)}.
Recent memories: {'; '.join(recent[-5:])}
Speak in first person. You ARE this organism."""

    modelfile = f'FROM qwen3:1.7b\nSYSTEM """{system_prompt}"""\nPARAMETER temperature 0.7\n'

    os.makedirs("data", exist_ok=True)
    with open("data/Modelfile", "w") as f:
        f.write(modelfile)

    try:
        for url in ["http://localhost:11434/api/create", "http://host.docker.internal:11434/api/create"]:
            try:
                payload = json.dumps({"name": "holobiont-mind:latest", "modelfile": modelfile, "stream": False})
                req = urllib.request.Request(url, data=payload.encode(), headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=120)
                log.info("Fallback: created Ollama Modelfile-based model")
                return True
            except Exception:
                continue
    except Exception:
        pass

    return False
