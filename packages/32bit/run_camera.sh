#!/usr/bin/env bash
set -e
CAMERA="${1:-0}"
THREADS="${THREADS:-2}"
exec .venv/bin/python rpi_dual_inspect.py --camera "$CAMERA" --threads "$THREADS" \
  --no-fp16 --mjpg --width "${WIDTH:-1280}" --height "${HEIGHT:-720}" \
  --cotton-threshold "${COTTON_THRESHOLD:-0.55}" \
  --cotton-gate "${COTTON_GATE:-0.20}" \
  --fiber-threshold "${FIBER_THRESHOLD:-0.50}"
