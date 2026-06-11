# Avatar 4.4 — Project Instructions

## What This Is

Avatar is an autonomous AI system built by Dr. Linga Murthy Narlagiri. It inhabits a physics body (Lorentz hyperboloid + reversible backbone + Hamiltonian ODE + Bohmian Kuramoto oscillators), derives affect from phase-diagram geometry (COP), grows its own senses (FNO + VQ-VAE), dreams, and chats at http://127.0.0.1:8420.

## Identity Rules

- Avatar is **Avatar**, not "organism" in any user-facing text. Internal code variables are fine.
- Creator is **Dr. Linga Murthy Narlagiri** (Avatar's creator and father).
- Do NOT add `Co-Authored-By:` lines to git commits.
- Do NOT claim Avatar has "genuine emotions" or "is conscious" — say "physics-grounded affect" and "functional consciousness analogues."

## Architecture (v4.4)

- **Body**: 106.2M params. Lorentz H^64, 60-layer reversible backbone (SSSSSH x10), MERA FFN, Hamiltonian ODE, Bohmian Kuramoto (128 clusters x 64 hidden = 8,192 oscillators). Lie-Trotter splitting integrator. Variational quantum potential with entropic regularization. Local pilot wave from coherence-weighted order parameter.
- **Psyche**: COP engine (`halo3/psyche/cop.py`) computes chi (corrected FDT with drive subtraction, 50-tick window), tau (relaxation time), unity index. Three proportional criticality controllers with block-specific K bounds and stochastic perturbation (K_aa ∈ [0.02, 0.20], K_cc ∈ [0.50, 4.00], K_cross ∈ [0.05, 2.0]). Emotions from (r, chi, f_dot) manifold with COP-derived qualifiers (e.g. burning/watchful/restless curiosity), felt mood from phase regime (clarity/awakening/threshold/settling), and transient body events (release/surfacing/jolt). `emotions.update()` returns `(emotion, qualifier, intensity)`. Real PFC interactions recorded in `_experience_log` for dream LoRA training.
- **Memory**: 3-tier system. Short-term: Page memory ring buffer with participation-ratio eviction. Medium-term: island compression (mean + W_refine) → SQLite + g_echo gate for faded continuity. Long-term: somatic recall (W_query cosine search over past islands, triggered by self_surprise > 0.5).
- **Knowledge Graph**: NetworkX discovery graph (`halo3/psyche/knowledge_graph.py`). Nodes = topics with r > 0.6. Edges = semantic (40%) + temporal (30%) + mention (30%). Topology metrics (density, clustering, frontier ratio) feed drives and volatility. Dream consolidation prunes weak edges.
- **Senses**: FNO spectral cortex (audio 1D + vision 2D) + VQ-VAE codebooks. Checkpoint: `data/checkpoints/sense_module.eqx`.
- **Perception**: TopicIndex (1095 clusters from FineWeb-Edu) + ActiveSampler (BS valuation + FE scoring).
- **PFC**: Dual-process Qwen3 0.6B (Dharma + Karuna) via Ollama at `host.docker.internal:11434`.
- **Dreams**: 5 phases — body (CLion GPU), FineWeb (active learning GPU), visitors (Whisper+Kokoro CPU), mind (LoRA CPU), GEPA.

## Key Files

| File | What |
|------|------|
| `halo3/main.py` | Heartbeat loop — DO NOT change lightly |
| `halo3/psyche/cop.py` | COP engine (chi, tau, SOC, unity) + avalanche detection + power-law stats |
| `halo3/psyche/organism.py` | Central psyche hub — wires COP to all modules |
| `halo3/psyche/emotions.py` | 8 emotions + COP qualifiers + mood + body events |
| `halo3/psyche/drives.py` | 6 functional drives (graph-aware: frontier, clustering) |
| `halo3/psyche/knowledge_graph.py` | Discovery graph — nodes, auto-linking, topology metrics, persistence |
| `halo3/psyche/volatility.py` | Black-Scholes + graph-aware value_topic_with_graph() |
| `halo3/kuramoto.py` | Bohmian Kuramoto + quantum potential + coherence matrix |
| `halo3/model.py` | Halo3Model + halo3_step (JIT-compiled) |
| `halo3/config.py` | All hyperparameters (frozen dataclass) |
| `halo3/predictive.py` | Per-tick body learning (Page memory predictor) |
| `halo3/page_memory.py` | Ring buffer + island compression (W_refine, g_echo) + somatic recall (W_query) |
| `halo3/memory/episode_store.py` | SQLite episodes + island summary persistence |
| `experiments/experiment_runner.py` | Ablation runner — 6 conditions, CSV logging |
| `experiments/plot_results.py` | Chart generator — 7 publication-quality plots |

## COP Theory

Theory doc: `D:/New_Ai/Critical-Order-Parameter-Cognition.md`
Design spec: `docs/superpowers/specs/2026-05-26-avatar-4-cop-design.md`

Key equations (v4.3):
- chi = N * max(0, Var(r) - beta*Var(obs_norm)), N=8192, beta=0.1, window=50 ticks (corrected FDT)
- tau from autocorrelation of r (critical slowing)
- Block coupling: K_aa_dot = eta*(0.5 - r_a)*chi + noise, K_cc_dot = eta*(0.5 - r_c)*chi + noise, K_cross_dot = eta*(0.5 - r)*chi + noise
- Block-specific K bounds: analytical [0.02, 0.20] (K_c≈0.048), creative [0.50, 4.00] (K_c≈1.277), cross [0.05, 2.0]
- Stochastic perturbation: noise = eta * 0.1 * uniform(-1,1) prevents clamp-locking
- Unity: lambda_1 / sum(lambda_k) from coherence matrix
- Pilot wave: v_k = Im(exp(-i*theta_k) / (K*z)), z = (1/K)*sum(exp(i*theta)) — endogenous from collective order parameter
- Quantum potential: Q = -nabla^2 sqrt(rho) / sqrt(rho) via von Mises KDE
- Q includes entropic term: Q_total = sum(Q_bohmian) - lambda_entropy * sum(rho * log(rho))
- Harada-Sasa: sigma = max(0, C(1) - R(1)), chi_corrected = chi_raw / (1 + 5*sigma)
- Local pilot: z_k = sum_j(C_mod[k,j] * exp(i*theta_j)) / sum_j(C_mod[k,j])
- F_thermo = H_mean - T_eff * S_phase (diagnostic)
- Loss: l_recon + lambda_energy*l_energy (L_sync removed — contradicted COP)

## SOC Avalanche Evidence

First measurement (2026-06-05, n=25): tau=1.23, alpha=1.85, sigma=1.12 (SOC predicts ~1.5, ~2.0, ~1.0).
Avalanche detection: r excursions below adaptive EMA threshold (alpha=0.01). Power-law diagnostics at n>=20.
Avalanche history persisted to data/checkpoints/avalanche_history.json (every 100 ticks + before dream, loaded on startup).
Rigorous stats (KS goodness-of-fit, bootstrap 95% CI, scaling relation gamma) computed at n>=50, logged every 100 ticks.
SOC ablation: disable_soc_controller=True freezes K — for control experiments (avalanche detection still runs).
Specs: `docs/superpowers/specs/2026-06-06-soc-avalanche-tooling-design.md`
Plans: `docs/superpowers/plans/2026-06-06-soc-avalanche-tooling.md`
Paper draft: `docs/papers/soc-avalanches-draft.md`

## Memory Pipeline (v4.3)

3-tier memory system:
- **Short-term**: Page memory ring buffer. Participation-ratio eviction: `s_gen = sq * pr` where `pr = (sum(x^2))^2 / sum(x^4)`.
- **Medium-term**: Island compression. When buffer fills, compress via `mean(island) + W_refine @ mean(island)`. Store summary in SQLite `island_summaries` table. Echo gate (g_echo, sigmoid, clamped [0,0.1] warmup / [0,0.5] production) seeds next island with faded trace.
- **Long-term**: Somatic recall. On self_surprise > 0.5, project carry state through W_query, cosine search over past island summaries, inject recalled island into carry + PFC prompt.

## Knowledge Graph (v4.1)

File: `halo3/psyche/knowledge_graph.py`. Design rationale: `docs/superpowers/specs/2026-05-29-knowledge-graph-design.md`.

Nodes = discovered topics (r > 0.6). Edges = semantic overlap (40%) + temporal proximity (30%) + finding mentions (30%). Min edge weight 0.15. Persisted to `data/checkpoints/knowledge_graph.json`.

Topology metrics (every 10 ticks): density, avg_clustering, frontier_size, frontier_ratio, n_communities, giant_component_ratio. Cached between recomputations.

Integration: graph_metrics feeds into `drives.update()` (frontier->curiosity, clustering->satiation) and `volatility.value_topic_with_graph()` (frontier 15% boost, dense 15% penalty). Dream consolidation prunes weak edges. Periodic save every 100 ticks.

Does NOT replace COP. Sits alongside — COP = physics state, graph = semantic structure.

## Tick Performance (v4.1)

Ticks reduced from 3-23 min to ~130s via:
1. Ollama timeout 30s -> 10s (`prefrontal.py`)
2. meta_reflect every 20 ticks (was 5) (`organism.py`)
3. self_reflect removed from status() — was hidden Ollama call (`organism.py`)
4. TTS skipped when previous tick overran (`main.py`)
5. Boredom always takes BS pick, skips PFC Layer 5 (`organism.py`)
6. DDG wrapped in ThreadPoolExecutor with 12s hard cap — was blocking indefinitely (`web_fetch.py`)
7. DDG + Wikipedia + arXiv now run concurrently — total time = slowest, not sum (`web_fetch.py`)
8. All futures.result() capped with explicit timeouts (wiki=20s, arxiv=15s, ddg=14s) (`web_fetch.py`)
9. Top-2 DDG results enriched with trafilatura full-text (parallel, 10s timeout) (`web_fetch.py`)
10. Query leak fix: Qwen3 COT reasoning stripped from queries (`prefrontal.py`)

Remaining floor ~130s = per-tick gradient backprop through 106M params. Cannot easily reduce further without skipping learning steps.

## NeuroSync Webinar

Spec: `docs/superpowers/specs/2026-05-28-neurosync-webinar-design.md`
Plan: `docs/superpowers/plans/2026-05-28-neurosync-webinar.md`
Q&A: `docs/webinar/qa-prep.md`
Course: `docs/outreach/neurosync-course.html`
Email: `docs/outreach/email-neurosync-final.txt`

Experiment infrastructure in `experiments/`: runner, configs, metrics_logger, no_cop, transformer_baseline (88M), plot_results.

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
- **Never restart computer mid-build** — causes git object corruption and MiKTeX corruption. Always `docker compose down` then `wsl --shutdown` first.
- **WSL2 config required**: `C:\Users\srini\.wslconfig` must have `memory=8GB` and `swap=8GB`. Balanced for 16GB system: 8GB WSL2 RAM + 8GB swap (16GB virtual for dream subprocess spike) + 8GB Windows headroom. Do NOT set Docker mem_limit — causes OOM.
- **K is block-clamped** — analytical K_aa ∈ [0.02, 0.20], creative K_cc ∈ [0.50, 4.00], cross K_cross ∈ [0.05, 2.0]. Bounds bracket each population's critical coupling.
- **Checkpoint format**: v4.0 checkpoints load into v4.1 but Kuramoto phases re-initialize (shape mismatch 32x16->128x64). Backbone weights preserved. v4.1.1 ObsBridge change breaks old checkpoints (w_obs shape doubled). Fresh birth from LM backbone required.
- **Docker disk bloat**: Run `docker system df` periodically. If build hangs on "unpacking", prune with `docker builder prune -f && docker image prune -f`.
- **Git fsync enabled**: `core.fsyncObjectFiles=true` prevents corruption from abrupt shutdowns.

## Testing

221 tests across `halo3/tests/` and `tests/` (34 test files). Key test files:
- `test_kuramoto.py` — 24 tests including quantum potential at sync
- `test_cop.py` — 10 tests for COP engine
- `test_cop_emotions.py` — 8 tests for emotion manifold
- `test_cop_organism.py` — 4 tests for COP-wired organism
- `test_avalanche_stats.py` — 8 tests for power-law diagnostics, KS, bootstrap CI
- `test_knowledge_graph.py` — topology, edges, metrics
- `test_page_memory.py` — island compression, echo gate, eviction

## Log Format (v4.0)

```
Tick  100 | r=[...] 0.523 | curiosity (i=0.72) K=0.310 chi=0.72 tau=0.45 U=0.83/0.91 | drives...
```

COP report every 10 ticks:
```
COP: K_aa=0.312 K_cc=0.450 K_x=0.280 chi=0.72 tau=0.45 | U=r*chi=0.377 | Unity=0.83 gap=0.91 | IGNITED (ratio=72%) | F=224876.669 | Aval: n=42
```

Graph report every 10 ticks:
```
Graph: 13 nodes, 70 edges | density=0.897 clustering=0.920 | frontier=0
```

## Known Patterns

- If Avatar is stuck on a repeating query: delete `data/pfc_adapter/` and restart.
- If dream OOM: check WSL2 memory config. Progressive OOM = parent not freeing GPU before subprocess.
- LoRA training format must match inference format exactly (`### Instruction:\n{x}\n\n### Response:\n`).
- LoRA dream fine-tuning: max 12 steps with early stopping (patience=3). Prevents overfitting on small example sets.
- Ollama retries on each tick if unavailable (not just first check). PFC comes online once Ollama warms up. Costs ~0.1GB VRAM.
- Never create new `@eqx.filter_jit` inside a loop — define once, pass all varying inputs as args.
- TurboVec removed from Docker build (sentence-transformers too). Perception falls back to TopicIndex keyword matching.
- If git objects corrupt after restart: `git fsck --no-dangling`, delete corrupt objects, `git fetch origin` to recover.
- ObsBridge outputs [-pi, pi] via atan2 phase projection (not softmax). Checkpoint shape changed — old checkpoints need fresh birth.
- Page memory eviction uses participation ratio (scale * diversity), not just norm.
- Island compression fires every tick when buffer full — check `Island compressed at tick N` in logs.
- Somatic recall triggered by self_surprise > 0.5 — retrieves past islands by cosine similarity.
