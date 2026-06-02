#!/usr/bin/env bash
# Always use the project venv (avoids conda/base uvicorn shadowing .venv/bin)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Missing .venv — create it first:"
  echo "  python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

exec "$ROOT/.venv/bin/python" -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
