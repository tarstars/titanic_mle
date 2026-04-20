#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Run 'uv sync --dev' first." >&2
  exit 1
fi

exec .venv/bin/python -m ipykernel install \
  --user \
  --name "titanic-mle" \
  --display-name "Python (titanic-mle)"
