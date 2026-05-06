# Three-Phase Nightly Dream Cycle — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single nightly LoRA session with a three-phase dream cycle that mimics human sleep-based learning — spaced repetition, curiosity-driven gap-filling, and scaffolded day-over-day knowledge building.

**Architecture:** A new `NightlyDreamCycle` orchestrator in `halo_fep/training/dream_cycle.py` runs three sequential phases within an expanded nightly window (02:00–03:00). A `LearningJournal` SQLite table tracks per-episode forgetting curves and per-day topic history. All existing components (`LoRATrainer`, `EpisodeStore`, `topic_bootstrap`, `hyperbolic_pretrain`) are reused unchanged.

**Tech Stack:** Python, SQLAlchemy, optax, JAX/Equinox, SM-2 spaced repetition algorithm

---

## 1. New Files

### `halo_fep/training/dream_cycle.py`
The `NightlyDreamCycle` class orchestrates three phases:

```
NightlyDreamCycle.run(model, key) -> (HaloFEPModel, dict)
  ├── Phase 1 — spaced_consolidation()   (~20 min)
  ├── Phase 2 — curiosity_gap_filling()  (~20 min)
  ├── Phase 3 — scaffolded_build()       (~15 min)
  └── LearningJournal.record(...)
```

Each phase is independent. If a phase raises an exception it is logged and skipped; the cycle continues. A `time.monotonic()` deadline check before each phase skips it with `status: "skipped_budget"` if less than 5 minutes remain in the nightly window.

### `halo_fep/memory/learning_journal.py`
`LearningJournal` backed by a SQLite table:

```sql
CREATE TABLE learning_journal (
    date           TEXT PRIMARY KEY,   -- "YYYY-MM-DD"
    trained_topics TEXT,               -- JSON list of cluster names trained
    phase1_loss    REAL,
    phase2_loss    REAL,
    phase3_loss    REAL,
    forgetting_state TEXT              -- JSON: {episode_id: {next_due: str, stability: float}}
)
```

Key methods:
- `get_due_episodes(store, today) -> list[Episode]` — returns episodes whose `next_due_date <= today`
- `update_forgetting_state(episode_ids, success: bool)` — increments stability and advances interval on success; does not advance on revert
- `get_yesterday_topics(today) -> list[str]` — returns `trained_topics` from the previous entry
- `record(date, trained_topics, log1, log2, log3)` — upserts one row

---

## 2. Modified Files

### `halo_fep/config.py`
New fields on `HaloFEPConfig`:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `nightly_duration_min` | `int` | `60` | Total nightly window length in minutes |
| `spaced_rep_intervals` | `list[int]` | `[1,3,7,14,30]` | SM-2 review intervals in days |
| `curiosity_fe_threshold` | `float` | `0.5` | Min free energy to count as a "gap encounter" |
| `curiosity_min_encounters` | `int` | `3` | Min gap encounters before a cluster triggers Phase 2 |
| `curiosity_since_days` | `int` | `7` | Lookback window for gap detection |
| `phase1_steps` | `int` | `150` | SGD steps in Phase 1 |
| `phase2_steps_per_cluster` | `int` | `200` | SGD steps per gap cluster in Phase 2 |
| `phase3_steps` | `int` | `100` | SGD steps in Phase 3 (matches current default) |
| `scaffold_boost` | `float` | `2.0` | Priority multiplier for yesterday's topics in Phase 3 |

### `halo_fep/main.py`
- Expand nightly window check from 15 min to `cfg.nightly_duration_min`
- Replace `LoRATrainer.run(get_high_confidence())` with `NightlyDreamCycle(cfg, store, journal, trainer).run(model, key)`
- Construct `LearningJournal` at startup (same pattern as `EpisodeStore`)

### `halo_fep/memory/episode_store.py`
Two additions:

**`add()` — topic tagging:** When storing a new episode, run a lightweight keyword match on the perception `query` string against `TOPIC_KEYWORDS` (already defined in `topic_bootstrap.py`) and store the matched cluster name in a new `topic_cluster TEXT` column. If no match, store `"general"`.

**`get_curiosity_gaps(since_days, min_encounters, fe_threshold) -> list[str]`:** Groups episodes from the last `since_days` by `topic_cluster`, counts how many had `free_energy > fe_threshold`, returns cluster names with `count >= min_encounters`, ordered by count descending.

---

## 3. Phase Details

### Phase 1 — Spaced Consolidation

- Calls `journal.get_due_episodes(store, today)` to get episodes whose spaced-repetition interval has elapsed
- Passes them to the existing `LoRATrainer.run()` (EWC-LoRA + revert-on-diverge)
- On success: `journal.update_forgetting_state(ids, success=True)` → stability++, next interval advances
- On revert: `journal.update_forgetting_state(ids, success=False)` → stability unchanged, same interval retried tomorrow
- New episodes (never reviewed) are enrolled into the forgetting curve the first time they appear in Phase 3

### Phase 2 — Curiosity Gap-Filling

- Calls `store.get_curiosity_gaps(since_days=cfg.curiosity_since_days, ...)` to get struggling topic clusters
- For each cluster (ordered by severity):
  - Wikipedia cluster → `iter_wikipedia_token_batches(cluster)` + `LoRATrainer.run()`
  - Semantic/relational cluster → `run_hyperbolic_pretrain()`
  - Unknown cluster → `run_bootstrap()` with `multiscale_strides=[1,4]`
- Budget: `cfg.phase2_steps_per_cluster` steps per cluster; clusters skipped if time budget exhausted
- Cluster-to-source mapping: Wikipedia clusters are the 8 keys from `TOPIC_KEYWORDS` in `topic_bootstrap.py`; all others route to hyperbolic or synthetic

### Phase 3 — Scaffolded Build

- Retrieves yesterday's `trained_topics` from the journal via `journal.get_yesterday_topics(today)`
- Calls `store.get_prioritized(n=200, since_timestamp=midnight, alpha=cfg.per_alpha, beta=cfg.per_beta)` with a `topic_boost` argument: episodes whose `topic_cluster` is in `yesterday_topics` have their raw priority multiplied by `cfg.scaffold_boost` before normalization
- `get_prioritized()` gains an optional `topic_boost: dict[str, float] | None` parameter
- Passes boosted episodes to `LoRATrainer.run()`
- All episodes successfully trained in Phase 3 are enrolled into the forgetting curve (added to `forgetting_state` with `stability=1.0`, `next_due = today + spaced_rep_intervals[0]` days)
- If no journal entry exists for yesterday (first run), falls back to plain PER with no boost

---

## 4. Safety & Budget Enforcement

- A `deadline = time.monotonic() + cfg.nightly_duration_min * 60 - 300` (5-min safety margin) is set at cycle start
- Before each phase: `if time.monotonic() > deadline: skip_phase(log "skipped_budget")`
- The existing revert-on-diverge in `LoRATrainer.run()` applies unchanged to Phases 1 and 3
- Phase 2 uses the same revert-on-diverge within `LoRATrainer.run()` for Wikipedia/synthetic paths; `run_hyperbolic_pretrain()` already has its own convergence guard
- If all three phases revert, the journal still records the attempt (with `status: "all_reverted"`) so forgetting curves and topic history remain consistent

---

## 5. Testing

- `halo_fep/memory/tests/test_learning_journal.py` — unit tests for SM-2 interval logic, `get_due_episodes`, `get_yesterday_topics`, `record`
- `halo_fep/training/tests/test_dream_cycle.py` — integration test: mock all three phases, verify orchestrator skips on budget exhaustion, verify journal is written
- `halo_fep/memory/tests/test_episode_store.py` — new tests for `get_curiosity_gaps` and `topic_cluster` tagging in `add()`
- `halo_fep/tests/test_config.py` — new fields present and validated
