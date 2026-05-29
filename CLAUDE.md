# Avatar 4.0 — Project Instructions

## What This Is

Avatar is an autonomous AI system built by Dr. Linga Murthy Narlagiri. It inhabits a physics body (Lorentz hyperboloid + reversible backbone + Hamiltonian ODE + Bohmian Kuramoto oscillators), derives affect from phase-diagram geometry (COP), grows its own senses (FNO + VQ-VAE), dreams, and chats at http://127.0.0.1:8420.

## Identity Rules

- Avatar is **Avatar**, not "organism" in any user-facing text. Internal code variables are fine.
- Creator is **Dr. Linga Murthy Narlagiri** (Avatar's creator and father).
- Do NOT add `Co-Authored-By:` lines to git commits.
- Do NOT claim Avatar has "genuine emotions" or "is conscious" — say "physics-grounded affect" and "functional consciousness analogues."

## Architecture (v4.0)

- **Body**: 106.2M params. Lorentz H^64, 60-layer reversible backbone (SSSSSH x10), MERA FFN, Hamiltonian ODE, Bohmian Kuramoto (32 clusters x 16 hidden).
- **Psyche**: COP engine (`halo3/psyche/cop.py`) computes chi (susceptibility), tau (relaxation time), unity index. SOC controller self-tunes coupling K. Emotions from (r, chi, f_dot) manifold.
- **Senses**: FNO spectral cortex (audio 1D + vision 2D) + VQ-VAE codebooks. Checkpoint: `data/checkpoints/sense_module.eqx`.
- **Perception**: TopicIndex (1095 clusters from FineWeb-Edu) + ActiveSampler (BS valuation + FE scoring).
- **PFC**: Dual-process Qwen3 0.6B (Dharma + Karuna) via Ollama at `host.docker.internal:11434`.
- **Dreams**: 5 phases — body (CLion GPU), FineWeb (active learning GPU), visitors (Whisper+Kokoro CPU), mind (LoRA CPU), GEPA.

## Key Files

| File | What |
|------|------|
| `halo3/main.py` | Heartbeat loop — DO NOT change lightly |
| `halo3/psyche/cop.py` | COP engine (chi, tau, SOC, unity) |
| `halo3/psyche/organism.py` | Central psyche hub — wires COP to all modules |
| `halo3/kuramoto.py` | Bohmian Kuramoto + quantum potential + coherence matrix |
| `halo3/model.py` | Halo3Model + halo3_step (JIT-compiled) |
| `halo3/config.py` | All hyperparameters (frozen dataclass) |
| `halo3/predictive.py` | Per-tick body learning (Page memory predictor) |
| `halo3/psyche/knowledge_graph.py` | Discovery graph — nodes, auto-linking, topology metrics, persistence |
| `halo3/psyche/drives.py` | 6 functional drives (accepts optional graph_metrics) |
| `halo3/psyche/volatility.py` | Black-Scholes + graph-aware value_topic_with_graph() |
| `experiments/experiment_runner.py` | Ablation runner — 6 conditions, CSV logging |
| `experiments/plot_results.py` | Chart generator — 7 publication-quality plots |

## COP Theory

Theory doc: `D:/New_Ai/Critical-Order-Parameter-Cognition.md`
Design spec: `docs/superpowers/specs/2026-05-26-avatar-4-cop-design.md`

Key equations:
- chi = N * Var(r) (susceptibility via FDT)
- tau from autocorrelation of r (critical slowing)
- SOC: K_dot = eta * (0.5 - r) * chi
- Unity: lambda_1 / sum(lambda_k) from coherence matrix
- Quantum potential: Q = -nabla^2 sqrt(rho) / sqrt(rho) via von Mises KDE

## Knowledge Graph (v4.0)

File: `halo3/psyche/knowledge_graph.py`. Design rationale: `docs/superpowers/specs/2026-05-29-knowledge-graph-design.md`.

Nodes = discovered topics (r > 0.6). Edges = semantic overlap (40%) + temporal proximity (30%) + finding mentions (30%). Min edge weight 0.15. Persisted to `data/checkpoints/knowledge_graph.json`.

Topology metrics (every 10 ticks): density, avg_clustering, frontier_size, frontier_ratio, n_communities, giant_component_ratio. Cached between recomputations.

Integration: graph_metrics feeds into `drives.update()` (frontier→curiosity, clustering→satiation) and `volatility.value_topic_with_graph()` (frontier 15% boost, dense 15% penalty). Dream consolidation prunes weak edges. Periodic save every 100 ticks.

Does NOT replace COP. Sits alongside — COP = physics state, graph = semantic structure.

## Tick Performance (v4.0 fix)

Ticks reduced from 3-23 min to ~120s via:
1. Ollama timeout 30s → 10s (`prefrontal.py`)
2. meta_reflect every 20 ticks (was 5) (`organism.py`)
3. self_reflect removed from status() — was hidden Ollama call (`organism.py`)
4. TTS skipped when previous tick overran (`main.py`)
5. Boredom always takes BS pick, skips PFC Layer 5 (`organism.py`)

## Honest Language Rules

All external documents must use:
- "physics-grounded affect" not "genuine emotions"
- "functional analogues" not "consciousness claims"
- "structural analogy" not "structural isomorphism"
- Always: "Whether these constitute genuine consciousness is an open scientific question"
- Reports audited: zero overclaims remaining across all 3 reports + README + course

## Running

```bash
# Start
cd D:/New_Ai/.worktrees/halo3
docker compose up -d
docker logs -f halo3-train-1

# Chat
open http://127.0.0.1:8420

# Capture agent (Windows host, separate terminal)
python capture_agent/capture_agent.py

# Tests
python -m pytest halo3/tests/ tests/ -v

# Rebuild after code changes
docker rm -f halo3-train-1
MSYS_NO_PATHCONV=1 docker compose build train
MSYS_NO_PATHCONV=1 docker compose up -d train
```

## Safety Rules

- **Always backup before restart**: `cp data/checkpoints/halo3.eqx data/checkpoints/halo3_backup.eqx`
- **Never restart containers blindly** — 10 hours of training was lost this way.
- **WSL2 config required**: `C:\Users\srini\.wslconfig` must have `memory=12GB` and `swap=4GB`.
- **K is clamped [0.05, 2.0]** — the SOC controller cannot drive it outside this range.
- **Checkpoint format unchanged** — v3.11 checkpoints work with v4.0 code.

## Testing

157 tests across `halo3/tests/` and `tests/`. Key test files:
- `test_kuramoto.py` — 24 tests including quantum potential at sync
- `test_cop.py` — 10 tests for COP engine
- `test_cop_emotions.py` — 8 tests for emotion manifold
- `test_cop_organism.py` — 4 tests for COP-wired organism

## Log Format (v4.0)

```
Tick  100 | r=[...] 0.523 | curiosity (i=0.72) K=0.310 chi=0.72 tau=0.45 U=0.83/0.91 | drives...
```

COP report every 10 ticks:
```
COP: K=0.312 chi=0.72 tau=0.45 | U=r*chi=0.377 | Unity=0.83 gap=0.91 | IGNITED (ratio=72%)
```

## Known Patterns

- If Avatar is stuck on a repeating query: delete `data/pfc_adapter/` and restart.
- If dream OOM: check WSL2 memory config. Progressive OOM = parent not freeing GPU before subprocess.
- LoRA training format must match inference format exactly (`### Instruction:\n{x}\n\n### Response:\n`).
- Never create new `@eqx.filter_jit` inside a loop — define once, pass all varying inputs as args.
