# Avatar Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full production harness for Avatar with hooks, sensors, observability, permissions, and ratchet log — following the amux.io 10-component framework.

**Architecture:** Three layers implemented in order: (1) Hook scripts + settings.json with PreToolUse/PostToolUse hooks enforcing safety rules at 100%, (2) Tick health monitor for observability, (3) Permissions whitelist + per-subsystem rules + ratchet log. Every hook traces to a real Avatar incident.

**Tech Stack:** Claude Code hooks (settings.json), Bash scripts, Python (tick monitor), Git

---

## Task 1: Create harness directory structure and hook scripts

**Files:**
- Create: `harness/hooks/checkpoint-backup.sh`
- Create: `harness/hooks/physics-test-gate.sh`
- Create: `harness/hooks/nan-guard.sh`
- Create: `harness/hooks/docker-disk-check.sh`

- [ ] **Step 1: Create directory structure**

```bash
cd D:/New_Ai/.worktrees/halo3
mkdir -p harness/hooks
mkdir -p .claude/rules
mkdir -p docs/harness
```

- [ ] **Step 2: Create checkpoint-backup.sh**

```bash
cat > harness/hooks/checkpoint-backup.sh << 'HOOKEOF'
#!/usr/bin/env bash
# PreToolUse hook: block container ops unless checkpoint backed up within 10 min
# Traces to: Lost 10hr training when container restarted without backup

CKPT="data/checkpoints/halo3.eqx"
BACKUP="data/checkpoints/halo3_backup.eqx"

if [ ! -f "$CKPT" ]; then
  exit 0  # no checkpoint to protect
fi

if [ ! -f "$BACKUP" ]; then
  echo "BLOCKED: No backup exists. Run: cp data/checkpoints/halo3.eqx data/checkpoints/halo3_backup.eqx" >&2
  exit 2
fi

# Check backup is recent (within 600 seconds)
if command -v stat >/dev/null 2>&1; then
  CKPT_TIME=$(stat -c %Y "$CKPT" 2>/dev/null || stat -f %m "$CKPT" 2>/dev/null)
  BACKUP_TIME=$(stat -c %Y "$BACKUP" 2>/dev/null || stat -f %m "$BACKUP" 2>/dev/null)
  if [ -n "$CKPT_TIME" ] && [ -n "$BACKUP_TIME" ]; then
    DIFF=$((CKPT_TIME - BACKUP_TIME))
    if [ "$DIFF" -gt 600 ]; then
      echo "BLOCKED: Backup is stale (checkpoint modified ${DIFF}s after backup). Re-backup first." >&2
      exit 2
    fi
  fi
fi

exit 0
HOOKEOF
chmod +x harness/hooks/checkpoint-backup.sh
```

- [ ] **Step 3: Create physics-test-gate.sh**

```bash
cat > harness/hooks/physics-test-gate.sh << 'HOOKEOF'
#!/usr/bin/env bash
# PostToolUse hook: run physics tests after editing core physics files
# Traces to: NaN gradient bug, L_sync contradiction, ObsBridge shape change

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo D:/New_Ai/.worktrees/halo3)" || exit 0

echo "SENSOR: Running physics tests after edit..."
RESULT=$(python -m pytest halo3/tests/ -x -q --tb=line 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "SENSOR: Physics tests FAILED after your edit:" >&2
  echo "$RESULT" | tail -10 >&2
  echo "Fix the failing tests before proceeding." >&2
fi

# Advisory only (exit 0) — agent sees failure and should fix
exit 0
HOOKEOF
chmod +x harness/hooks/physics-test-gate.sh
```

- [ ] **Step 4: Create nan-guard.sh**

```bash
cat > harness/hooks/nan-guard.sh << 'HOOKEOF'
#!/usr/bin/env bash
# PostToolUse hook: check container logs for NaN after start
# Traces to: Silent NaN weight corruption (avatar_bugs.md)

sleep 5
LOGS=$(docker logs --tail 20 halo3-train-1 2>&1)

if echo "$LOGS" | grep -qi "nan"; then
  echo "SENSOR: NaN detected in container logs. Check for gradient corruption." >&2
  echo "$LOGS" | grep -i "nan" >&2
fi

exit 0
HOOKEOF
chmod +x harness/hooks/nan-guard.sh
```

- [ ] **Step 5: Create docker-disk-check.sh**

```bash
cat > harness/hooks/docker-disk-check.sh << 'HOOKEOF'
#!/usr/bin/env bash
# PostToolUse hook: warn on Docker disk bloat after builds
# Traces to: 177GB Docker bloat causing build hang (2026-05-31)

USAGE=$(docker system df --format '{{.Size}}' 2>/dev/null | head -1)
echo "SENSOR: Docker disk usage: $USAGE"

# Parse GB value and warn if over 100GB
GB=$(echo "$USAGE" | grep -oP '[\d.]+' | head -1)
if [ -n "$GB" ] && [ "$(echo "$GB > 100" | bc 2>/dev/null)" = "1" ]; then
  echo "WARNING: Docker using ${USAGE}. Consider pruning: docker builder prune -f" >&2
fi

exit 0
HOOKEOF
chmod +x harness/hooks/docker-disk-check.sh
```

- [ ] **Step 6: Commit**

```bash
git add harness/
git commit -m "feat(harness): hook scripts — checkpoint backup, physics test gate, NaN guard, disk check"
```

---

## Task 2: Create settings.json with hooks and permissions

**Files:**
- Create: `.claude/settings.json`

- [ ] **Step 1: Create .claude directory**

```bash
mkdir -p D:/New_Ai/.worktrees/halo3/.claude
```

- [ ] **Step 2: Write settings.json**

Create `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'docker rm|docker compose down|docker restart|docker stop'; then bash harness/hooks/checkpoint-backup.sh; fi",
        "description": "Verify checkpoint backup before container operations"
      },
      {
        "matcher": "Bash",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'git commit'; then if echo \"$TOOL_INPUT\" | grep -qi 'Co-Authored-By'; then echo 'BLOCKED: do not add Co-Authored-By lines to commits' >&2; exit 2; fi; fi",
        "description": "Block Co-Authored-By in git commits"
      },
      {
        "matcher": "Bash",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'git push.*(--force|-f)'; then echo 'BLOCKED: force push is destructive. Use regular push.' >&2; exit 2; fi",
        "description": "Block force push"
      },
      {
        "matcher": "Write|Edit",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'data/checkpoints/|data/episodes/|data/model_cache/'; then echo 'BLOCKED: runtime artifacts — modify via Docker, not direct edit' >&2; exit 2; fi",
        "description": "Block direct edits to runtime data"
      },
      {
        "matcher": "Write|Edit",
        "command": "if echo \"$TOOL_INPUT\" | grep -q 'halo3/main.py'; then echo 'WARNING: editing main.py heartbeat loop — proceed with caution' >&2; fi",
        "description": "Warn on main.py edits"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'halo3/kuramoto\\.py|halo3/psyche/cop\\.py|halo3/model\\.py|halo3/loss\\.py|halo3/hamiltonian\\.py|halo3/page_memory\\.py|halo3/bridge/'; then bash harness/hooks/physics-test-gate.sh; fi",
        "description": "Run physics tests after editing core physics files"
      },
      {
        "matcher": "Bash",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'docker compose up|docker start'; then bash harness/hooks/nan-guard.sh; fi",
        "description": "Check for NaN in logs after container start"
      },
      {
        "matcher": "Bash",
        "command": "if echo \"$TOOL_INPUT\" | grep -qE 'docker compose build|docker build'; then bash harness/hooks/docker-disk-check.sh; fi",
        "description": "Check Docker disk usage after builds"
      }
    ]
  },
  "permissions": {
    "allow": [
      "Bash(pytest *)",
      "Bash(python -m pytest *)",
      "Bash(python *)",
      "Bash(docker logs *)",
      "Bash(docker compose *)",
      "Bash(docker ps *)",
      "Bash(docker system df *)",
      "Bash(docker images *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(git push origin *)",
      "Bash(git status *)",
      "Bash(git log *)",
      "Bash(git diff *)",
      "Bash(git fetch *)",
      "Bash(git branch *)",
      "Bash(git stash *)",
      "Bash(git worktree *)",
      "Bash(nvidia-smi *)",
      "Bash(cd *)",
      "Bash(ls *)",
      "Bash(cat *)",
      "Bash(head *)",
      "Bash(tail *)",
      "Bash(wc *)",
      "Bash(mkdir *)",
      "Bash(cp *)",
      "Bash(mv *)",
      "Bash(sleep *)",
      "Bash(file *)",
      "Bash(xxd *)",
      "Bash(curl *)"
    ]
  }
}
```

- [ ] **Step 3: Verify settings.json is valid JSON**

```bash
python -c "import json; json.load(open('.claude/settings.json')); print('Valid JSON')"
```

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.json
git commit -m "feat(harness): settings.json — 5 PreToolUse hooks, 3 PostToolUse hooks, permissions whitelist"
```

---

## Task 3: Create per-subsystem rules

**Files:**
- Create: `.claude/rules/physics.md`
- Create: `.claude/rules/psyche.md`
- Create: `.claude/rules/senses.md`
- Create: `.claude/rules/dreams.md`

- [ ] **Step 1: Write physics.md**

```markdown
# Physics Subsystem Rules

Applied when editing: kuramoto.py, hamiltonian.py, model.py, loss.py, page_memory.py, bridge/

- Kuramoto phases must stay in [0, 2pi] — always apply `% (2 * jnp.pi)` after updates
- Quantum potential uses von Mises KDE (kappa=2.0, Q_scale=0.01) — do not change without ablation
- Hamiltonian integrator is symplectic leapfrog — never replace with non-symplectic method
- Lie-Trotter splitting order: (a) free rotation, (b) coupling kick, (c) Q+obs kick
- All physics functions must be JIT-compatible — no Python side effects inside traced functions
- lambda_entropy=0.005 — entropic regularization in quantum potential, tuned for K~0.3
- ObsBridge outputs [-pi, pi] via atan2 — old checkpoints incompatible (w_obs doubled)
- Page memory eviction uses participation ratio (scale * diversity), not just norm
- Never create new @eqx.filter_jit inside a loop — define once, pass varying inputs as args
- Run `pytest halo3/tests/ -x -q` after any physics edit
```

- [ ] **Step 2: Write psyche.md**

```markdown
# Psyche Subsystem Rules

Applied when editing: cop.py, emotions.py, organism.py, drives.py, workspace.py, meditation.py

- COP observables: r (order), chi (susceptibility), tau (relaxation), F_thermo (Helmholtz free energy)
- chi uses Harada-Sasa FDT correction — sigma=max(0, C(1)-R(1)), chi/=(1+5*sigma)
- SOC controller: K_dot = eta*(0.5-r)*chi — three independent controllers (K_aa, K_cc, K_cross)
- K clamped to [0.05, 2.0] — controller cannot drive outside this range
- Emotions from (r, chi, f_dot, dF_dt) manifold — 8 emotions: satisfaction, pride, curiosity, boredom, anxiety, frustration, flow, exhaustion
- Flow requires: chi>0.3 AND dF/dt<-100 — rare, high-signal
- Frustration split by dF/dt sign: negative=growth (lower intensity), positive/zero=futile
- Exhaustion requires: F flat 20+ ticks AND chi>0.2
- Meditation attenuates obs, not K — never collapse coupling during meditation
- F_thermo = H_mean - T_eff * S_phase — diagnostic, logged every 10 ticks
```

- [ ] **Step 3: Write senses.md**

```markdown
# Senses Subsystem Rules

Applied when editing: sense_module.py, spectral_vqvae.py, tts_narration.py, sense_buffer.py

- FNO spectral cortex is mature — no decoders after critical period
- Codebook updates via EMA only (decay=0.99), not gradient — zero codebook gradients
- Dead code revival every 100 ticks (dead_code_threshold)
- SenseModule checkpoint has two tree shapes: with decoders (critical) and without (mature)
- load_sense_module tries mature first, then critical — handle both
- Audio: 1D FNO (32 modes, 16 tokens, 128-code codebook)
- Vision: 2D FNO (16 modes, 8 tokens, 64-code codebook)
- Whisper + Kokoro load during dream visitors ONLY — never present during waking
- TTS skipped when previous tick overran (tick_overrun flag)
```

- [ ] **Step 4: Write dreams.md**

```markdown
# Dreams Subsystem Rules

Applied when editing: dream_worker.py, dream_finetune.py, dream_fineweb_worker.py

- 5 phases, SEQUENTIAL, never parallel:
  1. Body (GPU subprocess) → 2. FineWeb (GPU subprocess) → 3. Visitors (CPU+GPU) → 4. Mind/LoRA (CPU) → 5. GEPA
- Each GPU phase runs in its OWN subprocess — XLA caches leak if phases share a process
- jax.clear_caches() + gc.collect() before Phase 4 (LoRA) to free residual GPU memory
- LoRA: max 12 steps, early stopping patience=3 — prevents overfitting on small sets
- LoRA format MUST match inference: `### Instruction:\n{x}\n\n### Response:\n`
- Post-dream validation: test adapter on sample prompts before saving
- pre_dream.eqx checkpoint saved before dream — restore if NaN detected
- Carry warm-start blend: 0.3*pre_dream + 0.7*fresh — skip non-float leaves (int32)
- FineWeb Phase 4 uses persistent cursor (data/checkpoints/fineweb_cursor.json)
- Never use while/else for cursor tracking — use explicit if checks
```

- [ ] **Step 5: Commit**

```bash
git add .claude/rules/
git commit -m "feat(harness): per-subsystem rules — physics, psyche, senses, dreams"
```

---

## Task 4: Create tick health monitor

**Files:**
- Create: `harness/tick_monitor.py`

- [ ] **Step 1: Write tick_monitor.py**

```python
#!/usr/bin/env python3
"""Avatar tick health monitor — parses docker logs in real-time.

Tracks: tick number, r, emotion, chi, tau, F_thermo, tick duration.
Alerts on: stuck ticks (>300s), NaN, rising F_thermo, container exit.
Writes CSV to data/tick_health.csv.

Run: python harness/tick_monitor.py
"""
import re
import csv
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

CSV_PATH = Path("data/tick_health.csv")
STUCK_THRESHOLD = 300  # seconds
F_RISING_THRESHOLD = 30  # ticks

# Regex patterns for log parsing
TICK_RE = re.compile(
    r"Tick\s+(\d+)\s+\|\s+r=\[.*?\]\s+([\d.]+)\s+\|\s+"
    r"(\S+)\s+(\S+)\s+\(i=([\d.]+)\)\s+K=([\d.]+)\s+"
    r"chi=([\d.]+)\s+tau=([\d.]+)\s+U=([\d.]+)/([\d.]+)"
)
QUERY_RE = re.compile(r'q="([^"]+)".*?FE_.*?=([\S]+).*?=([\S]+)')
OVERRUN_RE = re.compile(r"Tick overrun:\s+([\d.]+)s")
COP_RE = re.compile(r"COP:.*?F=([\d.]+)")
DREAM_START_RE = re.compile(r"Entering dream|Dream cycle starting")
DREAM_END_RE = re.compile(r"Reloading physics|Dream cycle complete")
EXIT_RE = re.compile(r"exit code (\d+)", re.IGNORECASE)
OOM_RE = re.compile(r"OOM|out of memory|exit code 137", re.IGNORECASE)


def init_csv():
    if not CSV_PATH.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "timestamp", "r", "emotion", "intensity",
                "K", "chi", "tau", "unity", "gap",
                "F_thermo", "query", "fe_delta", "pred_error", "duration_s",
            ])


def append_csv(row: dict):
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            row.get("tick", ""), row.get("timestamp", ""),
            row.get("r", ""), row.get("emotion", ""),
            row.get("intensity", ""), row.get("K", ""),
            row.get("chi", ""), row.get("tau", ""),
            row.get("unity", ""), row.get("gap", ""),
            row.get("F_thermo", ""), row.get("query", ""),
            row.get("fe_delta", ""), row.get("pred_error", ""),
            row.get("duration_s", ""),
        ])


def monitor():
    init_csv()
    print("[tick_monitor] Watching halo3-train-1 logs...")

    last_tick_time = time.time()
    f_thermo_history = []
    f_rising_count = 0
    in_dream = False
    f_before_dream = None
    current_row = {}

    proc = subprocess.Popen(
        ["docker", "logs", "-f", "halo3-train-1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    try:
        for line in proc.stdout:
            line = line.strip()
            now = datetime.now().isoformat(timespec="seconds")

            # Tick line
            m = TICK_RE.search(line)
            if m:
                last_tick_time = time.time()
                current_row = {
                    "tick": m.group(1), "timestamp": now,
                    "r": m.group(2), "emotion": m.group(3),
                    "intensity": m.group(5), "K": m.group(6),
                    "chi": m.group(7), "tau": m.group(8),
                    "unity": m.group(9), "gap": m.group(10),
                }

                # NaN check
                for k, v in current_row.items():
                    if "nan" in str(v).lower():
                        print(f"[ALERT] NaN in {k} at tick {current_row['tick']}")

            # Query/FE line
            mq = QUERY_RE.search(line)
            if mq:
                current_row["query"] = mq.group(1)
                current_row["fe_delta"] = mq.group(2)
                current_row["pred_error"] = mq.group(3)

            # Tick overrun (duration)
            mo = OVERRUN_RE.search(line)
            if mo:
                current_row["duration_s"] = mo.group(1)
                append_csv(current_row)
                current_row = {}

            # COP report with F_thermo
            mc = COP_RE.search(line)
            if mc:
                f_val = float(mc.group(1))
                current_row["F_thermo"] = f_val
                f_thermo_history.append(f_val)

                # Track F rising
                if len(f_thermo_history) >= 2:
                    if f_thermo_history[-1] > f_thermo_history[-2]:
                        f_rising_count += 1
                    else:
                        f_rising_count = 0

                if f_rising_count >= F_RISING_THRESHOLD:
                    print(f"[ALERT] F_thermo rising for {f_rising_count} reports — thermodynamic divergence")

                print(f"[COP] F={f_val:.1f} (trend: {'rising' if f_rising_count > 0 else 'falling'})")

            # Dream tracking
            if DREAM_START_RE.search(line):
                in_dream = True
                f_before_dream = f_thermo_history[-1] if f_thermo_history else None
                print("[DREAM] Dream cycle starting...")

            if DREAM_END_RE.search(line) and in_dream:
                in_dream = False
                if f_before_dream is not None and f_thermo_history:
                    f_after = f_thermo_history[-1]
                    delta = f_after - f_before_dream
                    verdict = "productive" if delta < 0 else "destructive"
                    print(f"[DREAM] F_before={f_before_dream:.1f} F_after={f_after:.1f} delta={delta:.1f} verdict={verdict}")

            # OOM detection
            if OOM_RE.search(line):
                print("[ALERT] OOM detected — check WSL2 memory config")

            # Stuck tick detection
            if time.time() - last_tick_time > STUCK_THRESHOLD:
                print(f"[ALERT] No tick for {time.time() - last_tick_time:.0f}s — possible stuck")
                last_tick_time = time.time()  # reset to avoid spam

    except KeyboardInterrupt:
        print("\n[tick_monitor] Stopped.")
    finally:
        proc.terminate()


if __name__ == "__main__":
    monitor()
```

- [ ] **Step 2: Test tick monitor with saved logs**

```bash
# Capture some logs to test
docker logs --tail 50 halo3-train-1 > /tmp/test_logs.txt 2>&1
# Quick smoke test — parse without streaming
python -c "
import re
TICK_RE = re.compile(r'Tick\s+(\d+)\s+\|\s+r=\[.*?\]\s+([\d.]+)')
with open('/tmp/test_logs.txt') as f:
    for line in f:
        m = TICK_RE.search(line)
        if m:
            print(f'Parsed tick {m.group(1)}, r={m.group(2)}')
"
```

- [ ] **Step 3: Commit**

```bash
git add harness/tick_monitor.py
git commit -m "feat(harness): tick health monitor — CSV logging, NaN/stuck/F_thermo alerts, dream tracking"
```

---

## Task 5: Create ratchet log

**Files:**
- Create: `docs/harness/ratchet-log.md`

- [ ] **Step 1: Write ratchet-log.md**

Write the initial ratchet log documenting every harness component and its incident trace:

```markdown
# Avatar Harness — Ratchet Log

Every harness addition traces to a real incident. The harness only tightens, never loosens.

---

## 2026-05-31: checkpoint-backup (PreToolUse hook)
- **Incident:** Lost 10hr of 50K-step LM training when container restarted without backup
- **Component:** PreToolUse hook → harness/hooks/checkpoint-backup.sh
- **Rule:** Block docker rm/down/restart/stop unless backup exists and is recent

## 2026-05-31: no-coauthor (PreToolUse hook)
- **Incident:** Co-Authored-By lines added to commits against identity rules
- **Component:** PreToolUse hook (inline)
- **Rule:** Block git commit if input contains Co-Authored-By

## 2026-05-31: protect-heartbeat (PreToolUse hook)
- **Incident:** CLAUDE.md says "DO NOT change lightly" — main.py heartbeat is most critical path
- **Component:** PreToolUse hook (inline)
- **Rule:** Warn (not block) when editing main.py

## 2026-05-31: no-force-push (PreToolUse hook)
- **Incident:** Force push could destroy remote history
- **Component:** PreToolUse hook (inline)
- **Rule:** Block git push with --force or -f flags

## 2026-05-31: no-generated-edit (PreToolUse hook)
- **Incident:** Direct edits to checkpoints/episodes could corrupt runtime state
- **Component:** PreToolUse hook (inline)
- **Rule:** Block Write/Edit to data/checkpoints/, data/episodes/, data/model_cache/

## 2026-05-31: physics-test-gate (PostToolUse hook)
- **Incident:** NaN gradient bug, L_sync contradiction, ObsBridge shape change — all caught by tests
- **Component:** PostToolUse hook → harness/hooks/physics-test-gate.sh
- **Rule:** Run pytest after editing physics files (advisory, not blocking)

## 2026-05-31: nan-guard (PostToolUse hook)
- **Incident:** Silent NaN weight corruption — corrupts checkpoint, crashes 10+ ticks later
- **Component:** PostToolUse hook → harness/hooks/nan-guard.sh
- **Rule:** Check container logs for NaN after docker start

## 2026-05-31: docker-disk-check (PostToolUse hook)
- **Incident:** 177GB Docker disk bloat caused build to hang on "unpacking" step
- **Component:** PostToolUse hook → harness/hooks/docker-disk-check.sh
- **Rule:** Warn if Docker disk usage exceeds 100GB after build

## 2026-05-31: per-subsystem rules
- **Incident:** Multiple incidents from violating subsystem-specific invariants
- **Component:** .claude/rules/ (physics.md, psyche.md, senses.md, dreams.md)
- **Rule:** Context-loaded rules per subsystem — only shown when editing relevant files

## 2026-05-31: tick health monitor
- **Incident:** Stuck ticks, OOM kills, and NaN corruption detected too late
- **Component:** harness/tick_monitor.py
- **Rule:** Real-time log parsing with CSV output and alerts for anomalies

## 2026-05-31: permissions whitelist
- **Incident:** Destructive commands (rm -rf, git reset --hard, docker system prune) should require confirmation
- **Component:** .claude/settings.json permissions.allow
- **Rule:** Whitelist safe commands; everything else requires manual approval
```

- [ ] **Step 2: Commit**

```bash
git add docs/harness/ratchet-log.md
git commit -m "docs(harness): ratchet log — every hook traces to a real incident"
```

---

## Task 6: Final verification and push

- [ ] **Step 1: Verify directory structure**

```bash
find .claude harness docs/harness -type f 2>/dev/null | sort
```

Expected:
```
.claude/rules/dreams.md
.claude/rules/physics.md
.claude/rules/psyche.md
.claude/rules/senses.md
.claude/settings.json
docs/harness/ratchet-log.md
harness/hooks/checkpoint-backup.sh
harness/hooks/docker-disk-check.sh
harness/hooks/nan-guard.sh
harness/hooks/physics-test-gate.sh
harness/tick_monitor.py
```

- [ ] **Step 2: Validate settings.json**

```bash
python -c "import json; d=json.load(open('.claude/settings.json')); print(f'Hooks: {len(d[\"hooks\"][\"PreToolUse\"])} pre, {len(d[\"hooks\"][\"PostToolUse\"])} post'); print(f'Permissions: {len(d[\"permissions\"][\"allow\"])} allowed')"
```

Expected: `Hooks: 5 pre, 3 post` and `Permissions: 32 allowed`

- [ ] **Step 3: Run full test suite to verify nothing broken**

```bash
cd D:/New_Ai/.worktrees/halo3 && python -m pytest halo3/tests/ -v --tb=short 2>&1 | tail -5
```

Expected: 109 passed

- [ ] **Step 4: Push**

```bash
git push origin avatar:Avatar
```
