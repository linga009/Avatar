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
