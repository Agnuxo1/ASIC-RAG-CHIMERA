#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEANDIR="${ROOT}/lean"

if ! command -v elan >/dev/null 2>&1; then
  echo "==> Installing elan (Lean toolchain manager)"
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
  export PATH="${HOME}/.elan/bin:${PATH}"
fi

echo "==> Lean toolchain"
cat "${LEANDIR}/lean-toolchain"

echo "==> lake build (strict: no_sorry + warningAsError)"
cd "${LEANDIR}"
lake update
lake build -- -Dno_sorry -DwarningAsError=true
