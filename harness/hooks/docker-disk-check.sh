#!/usr/bin/env bash
# PostToolUse hook: warn on Docker disk bloat after builds
# Traces to: 177GB Docker bloat causing build hang (2026-05-31)

USAGE=$(docker system df --format '{{.Size}}' 2>/dev/null | head -1)
echo "SENSOR: Docker disk usage: $USAGE"

exit 0
