#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_strict_smoke.sh path/to/screen-recording.mp4" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

uv run screen-agent diagnostics
uv run screen-agent models-check --strict
uv run screen-agent analyze "$1" --out runs/strict-smoke --sample-fps 1 --max-frames 30 --strict-models

