# Avatar Harness — Ratchet Log

Every harness addition traces to a real incident. The harness only tightens, never loosens.

---

## 2026-05-31: checkpoint-backup (PreToolUse hook)
- **Incident:** Lost 10hr of 50K-step LM training when container restarted without backup
- **Component:** PreToolUse hook -> harness/hooks/checkpoint-backup.sh
- **Rule:** Block docker rm/down/restart/stop unless backup exists and is recent

## 2026-05-31: no-coauthor (PreToolUse hook)
- **Incident:** Co-Authored-By lines added to commits against identity rules
- **Component:** PreToolUse hook (inline in settings.json)
- **Rule:** Block git commit if input contains Co-Authored-By

## 2026-05-31: protect-heartbeat (PreToolUse hook)
- **Incident:** CLAUDE.md says "DO NOT change lightly" — main.py heartbeat is most critical path
- **Component:** PreToolUse hook (inline in settings.json)
- **Rule:** Warn (not block) when editing main.py

## 2026-05-31: no-force-push (PreToolUse hook)
- **Incident:** Force push could destroy remote history
- **Component:** PreToolUse hook (inline in settings.json)
- **Rule:** Block git push with --force or -f flags

## 2026-05-31: no-generated-edit (PreToolUse hook)
- **Incident:** Direct edits to checkpoints/episodes could corrupt runtime state
- **Component:** PreToolUse hook (inline in settings.json)
- **Rule:** Block Write/Edit to data/checkpoints/, data/episodes/, data/model_cache/

## 2026-05-31: physics-test-gate (PostToolUse hook)
- **Incident:** NaN gradient bug, L_sync contradiction, ObsBridge shape change — all caught by tests
- **Component:** PostToolUse hook -> harness/hooks/physics-test-gate.sh
- **Rule:** Run pytest after editing physics files (advisory, not blocking)

## 2026-05-31: nan-guard (PostToolUse hook)
- **Incident:** Silent NaN weight corruption — corrupts checkpoint, crashes 10+ ticks later
- **Component:** PostToolUse hook -> harness/hooks/nan-guard.sh
- **Rule:** Check container logs for NaN after docker start

## 2026-05-31: docker-disk-check (PostToolUse hook)
- **Incident:** 177GB Docker disk bloat caused build to hang on "unpacking" step
- **Component:** PostToolUse hook -> harness/hooks/docker-disk-check.sh
- **Rule:** Warn if Docker disk usage is high after build

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
