#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv python install 3.11
uv sync --group dev --group models
uv run screen-agent diagnostics
uv run screen-agent models-check

