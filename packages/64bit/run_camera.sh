#!/usr/bin/env bash
set -e

CAMERA="${1:-0}"
THREADS="${THREADS:-2}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"

exec .venv/bin/python rpi_dual_inspect.py \
  --camera "$CAMERA" \
  --threads "$THREADS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --mjpg \
  --cotton-threshold "${COTTON_THRESHOLD:-0.46}" \
  --cotton-gate "${COTTON_GATE:-0.20}" \
  --fiber-threshold "${FIBER_THRESHOLD:-0.50}"
