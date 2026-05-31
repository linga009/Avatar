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
