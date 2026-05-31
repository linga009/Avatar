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
