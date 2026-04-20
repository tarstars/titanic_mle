#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8894}"
TOKEN="${TITANIC_MLE_JUPYTER_TOKEN:-titanic-mle-local-token}"

cd "$ROOT_DIR"

if [[ ! -x .venv/bin/jupyter-lab ]]; then
  echo "Missing .venv. Run 'uv sync --dev' first." >&2
  exit 1
fi

exec .venv/bin/jupyter-lab \
  --ServerApp.open_browser=False \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.port="$PORT" \
  --ServerApp.port_retries=0 \
  --IdentityProvider.token="$TOKEN" \
  --ServerApp.password='' \
  --ServerApp.disable_check_xsrf=False
