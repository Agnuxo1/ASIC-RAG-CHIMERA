#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "${ROOT}/scripts/setup_python.sh" ]]; then
  echo "missing ${ROOT}/scripts/setup_python.sh" >&2
  exit 2
fi

if [[ ! -d "${ROOT}/python/.venv" ]]; then
  "${ROOT}/scripts/setup_python.sh"
fi

# shellcheck disable=SC1090
source "${ROOT}/python/.venv/bin/activate"

echo "==> Python tests"
python -m pytest -q "${ROOT}/python/tests"

echo "==> Lean build (strict)"
"${ROOT}/scripts/setup_lean.sh"
