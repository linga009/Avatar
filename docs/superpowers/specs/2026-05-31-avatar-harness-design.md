# Avatar Harness Engineering — Full Design Spec

**Date:** 2026-05-31
**Principle:** Agent = Model + Harness (Hashimoto, Feb 2026)
**Methodology:** amux.io/guides/harness-engineering — 10-component framework
**Ratchet Rule:** Every hook/sensor traces to a real Avatar incident. Zero aspirational rules.

---

## Layer 1: Hooks & Sensors

### 1.1 PreToolUse Hooks (block before action)

All hooks go in `.claude/settings.json` under `hooks.PreToolUse`.

#### Hook: checkpoint-backup
- **Matcher:** Bash commands matching `docker rm|docker compose down|docker restart|docker stop`
- **Action:** Check if `data/checkpoints/halo3_backup.eqx` was modified within the last 600 seconds. If not, block with: `"BLOCKED: backup checkpoint first — cp data/checkpoints/halo3.eqx data/checkpoints/halo3_backup.eqx"`
- **Traces to:** Lost 10 hours of training when container was restarted without backup (feedback_checkpoint_safety.md)
- **Implementation:** Shell script `harness/hooks/checkpoint-backup.sh`

#### Hook: no-coauthor
- **Matcher:** Bash commands matching `git commit`
- **Action:** If `$TOOL_INPUT` contains `Co-Authored-By`, block with: `"BLOCKED: do not add Co-Authored-By lines to commits"`
- **Traces to:** Identity rule (feedback_no_coauthor.md)
- **Implementation:** Inline shell in settings.json

#### Hook: protect-heartbeat
- **Matcher:** Edit or Write to files matching `halo3/main.py`
- **Action:** If the edit touches lines containing `while True`, `tick_interval`, or `heartbeat`, warn: `"WARNING: editing heartbeat loop in main.py — this is the most critical code path. Proceed with extreme caution."`
- **Note:** Warning only (exit 0), not block (exit 2). The user may legitimately need to edit main.py.
- **Traces to:** CLAUDE.md rule "DO NOT change lightly"

#### Hook: no-force-push
- **Matcher:** Bash commands matching `git push`
- **Action:** If command contains `--force` or `-f`, block with: `"BLOCKED: force push is destructive. Use regular push."`
- **Traces to:** General safety — could destroy remote history

#### Hook: no-generated-edit
- **Matcher:** Edit or Write to files matching `data/checkpoints/|data/episodes/|data/model_cache/`
- **Action:** Block with: `"BLOCKED: these are runtime artifacts, not source code. Modify via Docker container, not direct edit."`
- **Traces to:** Checkpoint corruption risks

### 1.2 PostToolUse Hooks (validate after action)

All hooks go in `.claude/settings.json` under `hooks.PostToolUse`.

#### Hook: physics-test-gate
- **Matcher:** Edit or Write to files matching `halo3/kuramoto.py|halo3/psyche/cop.py|halo3/model.py|halo3/loss.py|halo3/hamiltonian.py|halo3/page_memory.py|halo3/bridge/`
- **Action:** Run `cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ -x -q --tb=line 2>&1 | tail -5`. If tests fail, output: `"SENSOR: physics tests failed after your edit. Fix before proceeding."`
- **Note:** Advisory (exit 0), not blocking. Agent sees the failure and should fix it.
- **Traces to:** NaN gradient bug, L_sync contradiction, ObsBridge shape change — all caught by tests

#### Hook: nan-guard
- **Matcher:** Bash commands matching `docker compose up|docker start`
- **Action:** After container starts, run `sleep 10 && docker logs --tail 5 halo3-train-1 2>&1 | grep -i nan`. If found, warn: `"SENSOR: NaN detected in container logs. Check for gradient corruption."`
- **Traces to:** Silent NaN weight corruption (avatar_bugs.md)

#### Hook: docker-disk-check
- **Matcher:** Bash commands matching `docker compose build|docker build`
- **Action:** After build, run `docker system df --format '{{.Size}}' | head -1`. Warn if total > 100GB.
- **Traces to:** 177GB Docker bloat causing build hang (2026-05-31)

### 1.3 Hook Scripts Directory

```
harness/
  hooks/
    checkpoint-backup.sh    # PreToolUse: verify backup exists before container ops
    physics-test-gate.sh    # PostToolUse: run tests after physics file edits
    nan-guard.sh            # PostToolUse: check logs for NaN after container start
    docker-disk-check.sh    # PostToolUse: warn on disk bloat after builds
```

---

## Layer 2: Observability

### 2.1 Tick Health Monitor

**File:** `harness/tick_monitor.py`
**Runs on:** Windows host, alongside capture_agent
**Input:** `docker logs -f halo3-train-1` (streaming)
**Output:** `data/tick_health.csv` + console alerts

**Parsed fields per tick:**
- tick_number, timestamp, r_mean, emotion, intensity, K, chi, tau, unity, gap
- query, fe_delta, prediction_error, tick_duration
- F_thermo (from COP report every 10 ticks)

**Alert conditions:**
| Condition | Threshold | Action |
|-----------|-----------|--------|
| Tick stuck | >300s since last tick | Print alert, optional notification |
| NaN detected | Any NaN in parsed fields | Print alert with field name |
| F_thermo rising | Positive dF/dt for 30+ ticks | Print warning: "thermodynamic divergence" |
| Container exit | Docker process stops | Print alert with exit code |
| OOM | Exit code 137 in logs | Print "OOM kill detected — check WSL2 memory" |

**CSV format:**
```
tick,timestamp,r,emotion,intensity,K,chi,tau,unity,F_thermo,query,fe_delta,pred_error,duration_s
1,2026-05-31T11:06:00,0.082,curiosity,0.57,0.250,0.50,0.50,1.00,,ai_research,0.0,33500,202.8
```

### 2.2 Dream Quality Tracker

**Integrated into tick_monitor.py** (not a separate script).
- Detects dream start: log line containing "Entering dream"
- Captures F_thermo from last COP report before dream
- Captures F_thermo from first COP report after dream
- Logs: `DREAM: F_before={x} F_after={y} delta={y-x} verdict={productive|destructive}`
- Productive = F_after < F_before (consolidation)
- Destructive = F_after > F_before (disorganization)

---

## Layer 3: Permissions & Orchestration

### 3.1 Permissions (settings.json)

Add to project-level `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(pytest*)",
      "Bash(python -m pytest*)",
      "Bash(docker logs*)",
      "Bash(docker compose up*)",
      "Bash(docker compose down*)",
      "Bash(docker compose build*)",
      "Bash(docker ps*)",
      "Bash(docker system df*)",
      "Bash(git add*)",
      "Bash(git commit*)",
      "Bash(git push origin*)",
      "Bash(git status*)",
      "Bash(git log*)",
      "Bash(git diff*)",
      "Bash(git fetch*)",
      "Bash(git branch*)",
      "Bash(nvidia-smi*)",
      "Bash(cd *)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(wc *)",
      "Bash(mkdir *)",
      "Bash(cp *)",
      "Bash(sleep *)"
    ]
  }
}
```

Destructive commands (`rm -rf`, `git reset --hard`, `docker system prune`, `git push --force`) are NOT in the allow list — they require manual confirmation.

### 3.2 Per-Subsystem Rules

```
.claude/rules/
  physics.md      # Rules for editing kuramoto.py, hamiltonian.py, model.py, loss.py
  psyche.md       # Rules for editing cop.py, emotions.py, organism.py, drives.py
  senses.md       # Rules for editing sense_module, spectral_vqvae, tts
  dreams.md       # Rules for editing dream_worker, dream_finetune, dream_fineweb
```

Each rule file is loaded only when the agent edits files in that subsystem. Contains subsystem-specific invariants (e.g., "kuramoto phases must stay in [0, 2pi]", "dream phases are sequential, never parallel").

### 3.3 Ratchet Log

**File:** `docs/harness/ratchet-log.md`

Every harness addition is logged with:
```markdown
## 2026-05-31: checkpoint-backup hook
- **Incident:** Lost 10hr training when container restarted without backup
- **Component:** PreToolUse hook
- **Rule:** Block docker rm/down/restart unless backup exists within 10 min
- **File:** harness/hooks/checkpoint-backup.sh
```

This is the institutional memory of what went wrong. The ratchet only tightens.

---

## File Structure

```
D:/New_Ai/.worktrees/halo3/
  .claude/
    settings.json           # Hooks + permissions (project-level)
    rules/
      physics.md            # Physics subsystem invariants
      psyche.md             # Psyche subsystem invariants
      senses.md             # Senses subsystem invariants
      dreams.md             # Dreams subsystem invariants
  harness/
    hooks/
      checkpoint-backup.sh  # PreToolUse: verify backup before container ops
      physics-test-gate.sh  # PostToolUse: run physics tests after edits
      nan-guard.sh          # PostToolUse: check for NaN in container logs
      docker-disk-check.sh  # PostToolUse: warn on disk bloat
    tick_monitor.py         # Observability: parse tick logs, CSV, alerts
  docs/
    harness/
      ratchet-log.md        # Every harness addition traces to an incident
```

---

## Testing the Harness

1. **Hook smoke tests:** Trigger each hook manually and verify it blocks/warns correctly
2. **Tick monitor:** Run against saved log file, verify CSV output and alert detection
3. **Permissions:** Verify allowed commands auto-approve, denied commands require confirmation
4. **Ratchet log:** Verify every hook entry has an incident reference
